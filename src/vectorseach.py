from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader
import os
import glob

# --- Robust Path Setup ---
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DB_LOCATION = os.path.join(SRC_DIR, '..', 'data', 'chroma_db')
DATA_DIR = os.path.join(SRC_DIR, '..', 'data')

# Simple setup
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Create or load vector store
vector_store = Chroma(
    collection_name="pdf_collection",
    embedding_function=embeddings,
    persist_directory=DB_LOCATION
)

# Check for new files to add
print("Checking for new files...")

# Get all files currently in data folder
all_files = glob.glob(os.path.join(DATA_DIR, "*.pdf")) + glob.glob(os.path.join(DATA_DIR, "*.txt"))

# Get files already in database
existing_docs = vector_store.get()
if existing_docs['metadatas']:
    # Just use filenames instead of full paths for comparison
    processed_filenames = set(os.path.basename(doc['source']) for doc in existing_docs['metadatas'])
else:
    processed_filenames = set()

# Find new files by comparing just the filenames
new_files = [f for f in all_files if os.path.basename(f) not in processed_filenames]

if new_files:
    print(f"Found {len(new_files)} new file(s) to process...")
    
    all_new_docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    for file_path in new_files:
        print(f"Processing {os.path.basename(file_path)}...")
        
        if file_path.endswith('.pdf'):
            # Load PDF files
            loader = PyPDFDirectoryLoader(os.path.dirname(file_path))
            docs = loader.load()
            # Filter to only the specific file we want
            docs = [doc for doc in docs if doc.metadata['source'] == file_path]
        else:
            # Load text files (like schedule.txt)
            loader = TextLoader(file_path)
            docs = loader.load()
        
        # Split into chunks
        split_docs = text_splitter.split_documents(docs)
        all_new_docs.extend(split_docs)
    
    # Add new documents to existing database
    vector_store.add_documents(documents=all_new_docs)
    print(f"Added {len(all_new_docs)} new chunks to database!")
else:
    print("No new files found. Database is up to date.")

# Create retriever for searching
retriever = vector_store.as_retriever()

