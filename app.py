import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Suppress TensorFlow oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize embeddings with explicit model name
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize ChromaDB
persist_directory = "datastore_db_new"
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 8})

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever, return_source_documents=True)
output_parser = StrOutputParser()

def create_prompt(context, question):
    """Create the prompt for the chatbot based on the conversation context and user question."""
    context_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in context])
    return [
        ("system", f"You are InfoBot, a helpful college chatbot. Your work is to provide CONCISE information. Be helpful with the user and provide help for their queries. Here is the conversation history:\n{context_str}"),
        ("user", f"Question: {question}")
    ]

def generate_response(context, question):
    """Generate response using the chatbot model."""
    prompt_template = ChatPromptTemplate.from_messages(create_prompt(context, question))
    response = qa_chain.invoke({"query": question})
    result = response["result"]
    source_doc = response["source_documents"][0]
    chunk_text = source_doc.page_content
    source = source_doc.metadata.get("source", "No source available")

    return f"**Answer:** {result}\n\n", result

def truncate_history(history, max_tokens=1000):
    """Truncate chat history to fit within the model's token limit."""
    truncated = []
    token_count = 0

    # Traverse history from the latest message backward
    for message in reversed(history):
        message_str = f"{message['role']}: {message['content']}"
        token_count += len(message_str.split())
        if token_count <= max_tokens:
            truncated.insert(0, message)  # Add to the start of the truncated history
        else:
            break
    return truncated

def handle_conversation():
    """Manage the chat interface and user interactions."""
    st.title("InfoBuddy")

    # Initialize conversation state
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Welcome to InfoBuddy! How can I help you?"}]
        st.session_state.evaluation_scores = []

    # Display chat history
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Handle user input
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        response, generated_text = generate_response(st.session_state.messages, prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

if __name__ == "__main__":
    handle_conversation()
