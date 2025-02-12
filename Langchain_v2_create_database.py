# from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai 
from dotenv import load_dotenv
import os
import shutil
from pymongo import MongoClient
# from langchain.embeddings import OpenAIEmbeddings

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
#---- Set OpenAI API key 
# Change environment variable name from "OPENAI_API_KEY" to the name given in 
# your .env file.
openai.api_key = os.environ['OPENAI_API_KEY']

CHROMA_PATH = "chroma"
DATA_PATH = "corpus/pdf"


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_mongodb(chunks)
    # save_to_chroma(chunks)


def load_documents():
    # Directory containing PDF and CSV files
    data_directory = "corpus"

    # Initialize a list to store all documents
    documents = []

    # Process PDF files
    print("Loading PDF documents...")
    pdf_directory = os.path.join(data_directory, "pdf")
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_directory, filename)
            loader = PyPDFLoader(file_path)  # Load PDF with metadata (filename, page number)
            documents.extend(loader.load())

    # Process CSV files
    # print("Loading CSV documents...")
    # csv_directory = os.path.join(data_directory, "csv")
    # for filename in os.listdir(csv_directory):
    #     if filename.endswith(".csv"):
    #         file_path = os.path.join(csv_directory, filename)
    #         loader = CSVLoader(file_path)  # Load CSV with metadata (filename, row number)
    #         documents.extend(loader.load())
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=500,
    length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")


    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

# MongoDB connection details
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "DrugWise"
COLLECTION_NAME = "document_embeddings"

def save_to_mongodb(chunks: list[Document]):
    # Connect to MongoDB
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # Clear the collection if it already exists
    collection.delete_many({})

    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings()

    # Process and store each chunk
    for i, chunk in enumerate(chunks, start=1):  # Use enumerate to track the index
        # Generate embedding for the chunk
        embedding = embeddings.embed_query(chunk.page_content)

        # Prepare the document to insert into MongoDB
        document = {
            "content": chunk.page_content,
            "metadata": chunk.metadata,
            "embedding": embedding
        }

        # Insert into MongoDB
        collection.insert_one(document)

        # Print status every 1000 chunks
        if i % 1000 == 0:
            print(f"{i} chunks have been inserted into MongoDB.")

    print(f"Saved {len(chunks)} chunks to MongoDB collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()