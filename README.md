# Website Chat Bot:

This project demonstrates a Retrieval-Augmented Generation (RAG) system that allows you to scrape content from a website, store its embeddings in Pinecone, and then use an OpenAI-powered chatbot to answer questions about the website's content.

# Features

- Web Scraping: Utilizes [Docling](https://pypi.org/project/docling/) to extract readable text content from any given URL.

- Data Processing: Uses [Polars](https://pola.rs/) for efficient handling and transformation of scraped data (though minimal in this simpler example).

- Text Embedding: Employs [sentence-transformers](https://www.sbert.net/) (specifically all-MiniLM-L6-v2) to convert text chunks into dense vector embeddings.

- Vector Storage: Leverages [Pinecone](https://www.pinecone.io/) (cloud service) as a scalable vector database to store and search embeddings.

- Large Language Model (LLM): Integrates with OpenAI's powerful language models (e.g., gpt-3.5-turbo) for generating human-like responses.

- RAG Architecture: Implements a RAG pipeline using [LangChain](https://www.langchain.com/) to retrieve relevant information from Pinecone before generating an answer, grounding the LLM's responses in the scraped content.

- Interactive UI: Provides a user-friendly chatbot interface built with [Streamlit](https://streamlit.io/).

  # Project Structure

```bash
website_chat_bot/
├── .env                        # Stores API keys (gitignore this!)
├── .streamlit/                 #hiddent streamlit folder where you can set ui apreance and security config.
│   └── config.toml             # Ui config toml.
├── app.py                      # Streamlit application (main entry point)
├── config.py                   # Configuration variables
├── web_scraper.py              # Functions for web scraping with Docling
├── embedder.py                 # Functions for embedding text
├── vector_db_manager.py        # Functions for interacting with Pinecone
├── requirements.txt            # Project dependencies
└── README.md
```

# Setup and Installation

1. Clone the Repository (or create the files manually):

```bash
git clone <your-repo-url>
cd rag_website_explainer
```

2. Create a Conda Environment:

```bash
conda create -n rag_env python=3.9 # Or your preferred Python version, e.g., 3.10, 3.11
conda activate rag_env
```

3. Install Dependencies:

```bash
conda env update --file environment.yml --prune
```

4. Configure API Keys:

- Sign up for a [Pinecone account](https://www.www.pinecone.io/) and get your API Key and Environment from your dashboard.

- Sign up for an [OpenAI account](https://platform.openai.com/) and get your API Key.

Create a file named `.env` in the root of your website_chat_bot directory (next to app.py) and add your keys:

```bash
# .env
PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
PINECONE_ENVIRONMENT="YOUR_PINECONE_ENVIRONMENT"
OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```

- Important: Do NOT commit your `.env` file to version control (e.g., Git). Add `.env` to your `.gitignore` file.

# How To Run

1. Activate your Conda environment:

```bash
conda activate rag_env
```

2. Run the Streamlit application:

```bash
streamlit run app.py
```

3. Access the Chatbot:
   Your web browser should automatically open to the Streamlit application (usually http://localhost:8501).

Usage

1. Enter a Website URL: In the Streamlit interface, type or paste the URL of the website you want the chatbot to explain (e.g., https://www.polars.tech/ or https://www.langchain.com/).

2. Load and Process: Click the "Load and Process Website" button. The application will:

3. Scrape the website content using Docling.

4. Chunk the text.

5. Generate embeddings for each chunk.

6. Store these embeddings and metadata in your Pinecone index.

7. Ask Questions: Once the processing is complete, you can start asking questions about the content of the loaded website in the chat input box. The chatbot will use RAG to find relevant information in Pinecone and generate an answer using OpenAI's LLM.

# Notes

- Pinecone Index Creation: The create_pinecone_index function will automatically create an index if it doesn't exist. Ensure PINECONE_INDEX_NAME in config.py is unique and follows Pinecone's naming conventions. For the free tier, ensure you choose a ServerlessSpec cloud and region that matches your Pinecone environment (e.g., cloud='aws', region='us-east-1' or cloud='gcp', region='us-central1').

- Docling Performance: Docling works well for many sites, but highly dynamic, JavaScript-heavy sites might require more advanced scraping techniques (e.g., headless browsers like Playwright).

- Chunking Strategy: The current chunking is basic (splitting by double newlines). For production RAG, explore LangChain's RecursiveCharacterTextSplitter for more sophisticated and robust text splitting.

- Embedding Model: all-MiniLM-L6-v2 is a good general-purpose embedding model. For specific domains, you might consider fine-tuned or larger models.

- Cost: Be mindful of your OpenAI API usage, as costs can accrue based on token usage. Monitor your OpenAI dashboard. Pinecone's free tier has generous limits but also monitor your usage.

- Enjoy building your RAG website explainer!
- You can use the blueprint for this setup to create your own RAG chat bots related to documents or emails.
