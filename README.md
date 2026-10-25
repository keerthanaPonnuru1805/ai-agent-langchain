# AI Agent with Tools and Memory

A conversational AI agent built with LangChain, LangGraph and Groq (Llama 3.3).

## Features
- Tool calling: calculator, current date/time, Wikipedia lookup
- Conversation memory using LangGraph MemorySaver
- Secure math evaluation (no eval)

## Setup
1. Get a free API key at https://console.groq.com
2. Create a file named `.env` in this folder:
   GROQ_API_KEY=your_key_here
3. Install and run:
   pip install -r requirements.txt
   python agent.py

## Try these
- What is 15% of 2480?
- Tell me about Retrieval-Augmented Generation
- What day is it today?
- Add 100 to the first number I asked about   (tests memory)
