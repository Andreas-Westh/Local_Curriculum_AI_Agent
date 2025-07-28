from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
import os

# --- Robust Path Setup ---
# Get the absolute path of the directory this file is in (i.e., .../src)
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct the absolute path to the data and db directories
DB_LOCATION = os.path.join(SRC_DIR, '..', 'data', 'chroma_db')
DATA_DIR = os.path.join(SRC_DIR, '..', 'data')


# Simple setup
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Check if we need to load documents
add_documents = not os.path.exists(DB_LOCATION)

if add_documents:
    print("Loading PDFs from data folder...")
    
    # Load all PDFs from the absolute data directory path
    loader = PyPDFDirectoryLoader(DATA_DIR)
    documents = loader.load()
    
    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(documents)
    
    print(f"Loaded {len(docs)} chunks from PDFs")

# Create or load vector store
vector_store = Chroma(
    collection_name="pdf_collection",
    embedding_function=embeddings,
    persist_directory=DB_LOCATION
)

# Add documents if this is first time
if add_documents:
    vector_store.add_documents(documents=docs)
    print("Documents added to vector store!")

# Create retriever for searching
retriever = vector_store.as_retriever(
    #search_kwargs={"k": 3}
)

