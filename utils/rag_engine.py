import os
import glob
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# Ensure chroma db dir exists
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

# Define embedding model - using the free HuggingFace model as specified
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_documents():
    """Load all PDF and TXT files from the data directory."""
    documents = []
    
    # Load PDFs
    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")
            
    # Load TXTs
    txt_files = glob.glob(os.path.join(DATA_DIR, "*.txt"))
    for txt_file in txt_files:
        try:
            loader = TextLoader(txt_file, encoding='utf-8')
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {txt_file}: {e}")
            
    return documents

def get_vector_store():
    """Initialize or load the ChromaDB vector store."""
    vector_store = Chroma(
        collection_name="aircargo_rules",
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    
    try:
        # If the vector store is empty, load and index the documents
        if vector_store._collection.count() == 0:
            print("Vector store is empty. Loading and indexing documents... This may take a moment.")
            docs = load_documents()
            
            if not docs:
                print("No documents found in the data directory!")
                return vector_store
                
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=100
            )
            splits = text_splitter.split_documents(docs)
            
            # Add to vector store
            vector_store.add_documents(splits)
            print(f"Successfully indexed {len(splits)} chunks from {len(docs)} document pages.")
    except Exception as e:
        print(f"Error checking or indexing vector store: {e}")
        
    return vector_store

# Initialize a global instance so it's loaded once and ready for the agents
vector_store = get_vector_store()

def search_aircargo_rules(query: str, k: int = 4) -> str:
    """
    Search the vector store for the given query and return formatted results.
    This function will be used as a Tool by the Logistics Researcher Agent.
    """
    results = vector_store.similarity_search(query, k=k)
    
    if not results:
        return "No relevant information found in the knowledge base."
        
    formatted_results = []
    for i, doc in enumerate(results):
        source = os.path.basename(doc.metadata.get('source', 'Unknown source'))
        page = doc.metadata.get('page', 'Unknown page')
        # Clean up newlines for better LLM readability
        content = doc.page_content.replace('\n', ' ')
        formatted_results.append(f"--- Result {i+1} ---\nSource: {source} (Page {page})\nContent: {content}\n")
        
    return "\n".join(formatted_results)
