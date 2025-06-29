# rag_website_explainer/app.py
import streamlit as st
import polars as pl
from langchain_openai import ChatOpenAI # For OpenAI LLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# Import functions from our local scripts
from web_scraper import get_website_content_as_polars_df
from embedder import get_embedding_model
from vector_db_manager import get_pinecone_client, create_pinecone_index, add_documents_to_pinecone, get_pinecone_retriever
from config import OPENAI_API_KEY, OPENAI_MODEL_NAME, EMBEDDING_MODEL_NAME, PINECONE_INDEX_NAME, EMBEDDING_DIMENSION


# --- Global Initialization (Run once and cache) ---
@st.cache_resource
def initialize_components():
    """Initializes LLM, Embedding Model, and Pinecone Client/Index."""
    # Initialize LLM (OpenAI)
    llm = None
    if OPENAI_API_KEY:
        try:
            llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=OPENAI_MODEL_NAME, temperature=0.0)
            # Test if LLM is responsive (optional but good for debugging)
            llm_test_response = llm.invoke("Hello, who are you?")
            print(f"OpenAI LLM Test: {llm_test_response.content[:50]}...")
            st.success("Connected to OpenAI LLM!")
        except Exception as e:
            st.error(f"Error connecting to OpenAI LLM. Please check your API key and model name. Error: {e}")
    else:
        st.error("OpenAI API key not found in .env. Please set OPENAI_API_KEY.")

    # Initialize Embedding Model
    embedding_model = get_embedding_model(EMBEDDING_MODEL_NAME)
    if not embedding_model:
        st.error(f"Could not load embedding model: {EMBEDDING_MODEL_NAME}. Check internet connection for download or model name.")

    # Initialize Pinecone Client and Index
    pc_client = get_pinecone_client()
    pinecone_index = None
    if pc_client and embedding_model:
        # Create or get the Pinecone index. Ensure dimension matches embedding model.
        try:
            pinecone_index = create_pinecone_index(pc_client, PINECONE_INDEX_NAME, EMBEDDING_DIMENSION)
            st.success(f"Connected to Pinecone index: '{PINECONE_INDEX_NAME}'")
        except Exception as e:
            st.error(f"Error creating/connecting to Pinecone index: {e}. Check Pinecone environment and API key.")
    elif not pc_client:
        st.error("Pinecone client not initialized. Check PINECONE_API_KEY and PINECONE_ENVIRONMENT in .env.")
    
    return llm, embedding_model, pc_client, pinecone_index

# Call the initialization function
llm, embedding_model, pc_client, pinecone_index = initialize_components()

# --- Streamlit UI ---
st.set_page_config(page_title="Website Explainer Chatbot (Pinecone + OpenAI)", layout="wide")
st.title("🌐 Website Explainer Chatbot")
st.write("Enter a website URL, load its content, and then ask questions about it!")

# Check if all critical components are initialized
components_ready = llm and embedding_model and pc_client and pinecone_index

if not components_ready:
    st.warning("Some core components failed to initialize. Please check console for errors and your .env file.")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_website_loaded" not in st.session_state:
    st.session_state.current_website_loaded = None

# Input for URL
website_url = st.text_input("Enter the URL of the website to explain:", key="url_input")

if st.button("Load and Process Website"):
    if not components_ready:
        st.error("Cannot process website: Core components are not ready.")
    elif website_url:
        with st.spinner(f"Scraping and processing {website_url}... This might take a moment."):
            df_content = get_website_content_as_polars_df(website_url)
            if not df_content.is_empty():
                documents = [
                    Document(page_content=row["text"], metadata={"source_url": row["source_url"], "chunk_id": row["chunk_id"]})
                    for row in df_content.iter_rows(named=True)
                ]
                try:
                    add_documents_to_pinecone(pinecone_index, documents, embedding_model)
                    st.success(f"Successfully loaded and processed {len(documents)} chunks from {website_url} into Pinecone!")
                    st.session_state.current_website_loaded = website_url
                    st.session_state.messages.append({"role": "assistant", "content": f"I've loaded the content from {website_url}. Ask me anything about it!"})
                except Exception as e:
                    st.error(f"Failed to add documents to Pinecone: {e}. Ensure your Pinecone index is correctly configured and accessible.")
            else:
                st.error(f"Could not scrape any content from {website_url}. Please check the URL or try another one. Docling might struggle with highly dynamic sites.")
    else:
        st.warning("Please enter a URL to load.")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask me about the website..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        if not components_ready:
            st.error("Cannot answer questions: Core components are not ready. Please resolve the errors above.")
            st.session_state.messages.append({"role": "assistant", "content": "Error: Chatbot components are not fully initialized."})
        elif st.session_state.current_website_loaded is None:
            st.warning("Please load a website first before asking questions.")
            st.session_state.messages.append({"role": "assistant", "content": "Please load a website first before asking questions."})
        else:
            # Get the Pinecone retriever instance
            retriever = get_pinecone_retriever(pc_client, PINECONE_INDEX_NAME, embedding_model)
            
            # Define prompt for LLM
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful assistant that explains the content of a website.
                 Use the following retrieved context to answer the user's question accurately and concisely.
                 If you cannot find the answer in the provided context, politely state that you don't have enough information from the *provided website content*.
                 Do not make up answers.
                 
                 Context: {context}"""),
                ("human", "{input}")
            ])

            # LangChain RAG Chain
            # create_stuff_documents_chain combines documents into the prompt
            question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
            
            # create_retrieval_chain combines retriever and question_answer_chain
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)

            try:
                with st.spinner("Thinking..."):
                    # Invoke the RAG chain
                    response = rag_chain.invoke({"input": prompt})
                
                assistant_response = response["answer"]
                st.markdown(assistant_response)
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            except Exception as e:
                st.error(f"An error occurred while generating the response: {e}")
                st.session_state.messages.append({"role": "assistant", "content": "I encountered an error trying to answer that. Please try again."})
