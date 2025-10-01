A conversational AI assistant built with LangChain, NVIDIA AI Endpoints, and Gradio to help customers with  inquiries and support.
Overview

This application creates an interactive chat interface that:

Helps customers look up their part information using their personal details
Maintains conversation context to provide relevant responses
Extracts and tracks key information throughout the conversation
Delivers a user-friendly interface via Gradio

Features

Customer Identification: Identifies customers by first name, last name, and confirmation number
Information Retrieval: Retrieves part information from a knowledge base
Context Tracking: Maintains conversation history and tracks unresolved issues
User-Friendly Interface: Simple chat interface built with Gradio

Dependencies

langchain
langchain-nvidia-ai-endpoints
gradio
pydantic
python-dotenv
