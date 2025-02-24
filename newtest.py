from langchain.schema.runnable import (
    RunnableBranch,
    RunnableLambda,
    RunnableMap,       ## Wrap an implicit "dictionary" runnable
    RunnablePassthrough,
)

from langchain.schema.runnable.passthrough import RunnableAssign
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, SystemMessage, ChatMessage, AIMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser   
from typing import Dict, Union, Optional, List, Tuple
from dotenv import load_dotenv
from operator import itemgetter
import os
import gradio as gr

load_dotenv()

def get_flight_info(d: dict) -> str:
    """
    Example of a retrieval function which takes a dictionary as key. Resembles SQL DB Query
    """
    req_keys = ['first_name', 'last_name', 'confirmation']
    assert all((key in d) for key in req_keys), f"Expected dictionary with keys {req_keys}, got {d}"

    ## Static dataset. get_key and get_val can be used to work with it, and db is your variable
    keys = req_keys + ["part#", "name"]
    values = [
        ["Jane", "Doe", 12345, "P123", "Cummins Engine"],
        ["John", "Smith", 54321, "P456", "Cummins Filter"],
        ["Alice", "Johnson", 98765, "P789", "Wheel Hub"],
        ["Bob", "Brown", 56789, "P012", "Water Pump"],
    ]
    get_key = lambda d: "|".join([d['first_name'], d['last_name'], str(d['confirmation'])])
    get_val = lambda l: {k:v for k,v in zip(keys, l)}
    db = {get_key(get_val(entry)) : get_val(entry) for entry in values}

    # Search for the matching entry
    data = db.get(get_key(d))
    if not data:
        return (
            f"Based on {req_keys} = {get_key(d)}) from your knowledge base, no info on the user flight was found."
            " This process happens every time new info is learned. If it's important, ask them to confirm this info."
        )
    return (
        f"Found part information for {data['first_name']} {data['last_name']}: "
        f"Part #{data['part#']} - {data['name']}"
    )


def get_key_fn(base: BaseModel) -> dict:
    '''Given a dictionary with a knowledge base, return a key for get_flight_info'''
    return {  ## More automatic options possible, but this is more explicit
        'first_name' : base.first_name,
        'last_name' : base.last_name,
        'confirmation' : base.confirmation,
    }

instruct_chat = ChatNVIDIA(
    model="mistralai/mixtral-8x7b-instruct-v0.1",
    api_key=os.getenv("NVIDIA_API_KEY"),
) 

instruct_chat_llm = ChatNVIDIA(
    model="mistralai/mixtral-8x7b-instruct-v0.1",
    api_key=os.getenv("NVIDIA_API_KEY"),
)

instruct_llm = instruct_chat | StrOutputParser()

chat_model =  instruct_chat_llm | StrOutputParser()


external_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a chatbot for Power Truck Group Truck Parts and equipment dealer, and you are helping a customer with their issue."
        " Please chat with them! Stay concise and clear!"
        " \nUsing that, we retrieved the following: {context}\n"
        " If they provide info and the retrieval fails, ask to confirm their first/last name and confirmation number."
        " Do not ask them any other personal info."
        " The checking happens automatically; you cannot check manually."
    )),
    ("user", "{input}"),
])

class KnowledgeBase(BaseModel):
    first_name: str = Field('unknown', description="Chatting user's first name, `unknown` if unknown")
    last_name: str = Field('unknown', description="Chatting user's last name, `unknown` if unknown")
    confirmation: int = Field(0, description="order Confirmation Number, `0` if unknown")
    discussion_summary: str = Field("", description="Summary of discussion so far, including , issues, etc.")
    open_problems: list = Field([], description="Topics that have not been resolved yet")
    current_goals: list = Field([], description="Current goal for the agent to address")


    model_config = {
        "extra": "allow"  # Allow extra fields in the model
    }

parser_prompt = ChatPromptTemplate.from_template(
    "You are a chat assistant representing the truck parts dealer Power Truck Group, and are trying to track info about the conversation."
    " You have just received a message from the user. Please fill in the schema based on the chat."
    "\n\n{format_instructions}"
    "\n\nOLD KNOWLEDGE BASE: {know_base}"
    "\n\nASSISTANT RESPONSE: {output}"
    "\n\nUSER MESSAGE: {input}"
    "\n\nNEW KNOWLEDGE BASE: "
)

fail_str = (
    "You cannot access user's order details until they provide "
    "first name, last name, and order confirmation number."
)



def RExtract(pydantic_class, llm, prompt):
    '''
    Runnable Extraction module
    Returns a knowledge dictionary populated by slot-filling extraction
    '''
    parser = PydanticOutputParser(pydantic_object=pydantic_class)
    instruct_merge = RunnableAssign({'format_instructions' : lambda x: parser.get_format_instructions()})
    def preparse(string):
        # If we get an empty or invalid response, return default values
        if not string or '{' not in string:
            return pydantic_class()
            
        # Clean up the string - fix escape sequences
        string = (string
            .replace("\\_", "_")
            .replace("\n", " ")
            .replace("]", "]")  # removed invalid escape
            .replace("[", "[")  # removed invalid escape
        )
        return string
    return instruct_merge | prompt | llm | preparse | parser

external_chain = external_prompt | chat_model

extractor = RunnableAssign({'know_base' : RExtract(KnowledgeBase, instruct_llm, parser_prompt)})
internal_chain = extractor | RunnableAssign(
    {'context' : lambda d: get_flight_info(get_key_fn(itemgetter('know_base')(d)))}
)

state = {'know_base' : KnowledgeBase()}


def chat_gen(message, history=[], return_buffer=False):
    
    global state

    input_state = {
        'input': message,
        'history': history,
        'output': "" if not history else history[-1][1] if history[-1][1] else "",
        'know_base': state['know_base']
    }

    try:
        # Run the internal chain first (non-streaming)
        state = internal_chain.invoke(input_state)
        print(state)
        # Then stream the external response
        response = ""
        for chunk in external_chain.stream(state):
            response += chunk
            yield response
    except Exception as e:
        print(f"Error in chat_gen: {str(e)}")
        yield "I apologize, but I encountered an error. Please try again."

chatbot = gr.Chatbot(value = [[None, "Hello! I'm your Power Truck Group assistant! How can I help you?"]])
demo = gr.ChatInterface(chat_gen, chatbot=chatbot).queue()

try:
    ## NOTE: This should also give you a temporary public link which can be
    ## used to access this info on the public web while the session is live.
    demo.launch(debug=True, share=True, show_api=False)
    demo.close()
except Exception as e:
    demo.close()
    print(e)
    raise e

