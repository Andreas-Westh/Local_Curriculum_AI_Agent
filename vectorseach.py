from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
import os

# Simple setup
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
db_location = "data/chroma_db"

# Check if we need to load documents
add_documents = not os.path.exists(db_location)

if add_documents:
    print("Loading PDFs from data folder...")
    
    # Load all PDFs from data directory
    loader = PyPDFDirectoryLoader("data/")
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
    persist_directory=db_location
)

# Add documents if this is first time
if add_documents:
    vector_store.add_documents(documents=docs)
    print("Documents added to vector store!")

# Create retriever for searching
retriever = vector_store.as_retriever(
    #search_kwargs={"k": 3}
                                      )

