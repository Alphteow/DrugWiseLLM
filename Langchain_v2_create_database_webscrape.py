import pandas as pd
import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
import openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

# MongoDB connection details
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "DrugWise"
COLLECTION_NAME = "document_embeddings"

def fetch_article_content(url):
    """
    Fetch the content of an article from the given URL.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the main content of the article (this may vary depending on the website structure)
        # For PubMed articles, the abstract is usually in a specific tag
        abstract = soup.find('div', {'class': 'abstract-content'})
        if abstract:
            return abstract.get_text(strip=True)
        else:
            print(f"Could not find abstract for URL: {url}")
            return None
    except Exception as e:
        print(f"Error fetching URL {url}: {e}")
        return None

def save_to_mongodb(data):
    """
    Save the data (content, metadata, embeddings) to MongoDB.
    """
    # Connect to MongoDB
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # Clear the collection if needed (optional)
    # collection.delete_many({})

    # Insert the data into MongoDB
    collection.insert_one(data)

def process_csv_and_store_embeddings(csv_file):
    """
    Process the CSV file, fetch article content, generate embeddings, and store in MongoDB.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings()

    # Iterate through the 'url' column
    for index, row in df.iterrows():
        url = row['url']
        print(f"Processing URL: {url}")

        # Fetch the article content
        content = fetch_article_content(url)
        if not content:
            continue  # Skip if content could not be fetched

        # Generate embeddings for the content
        embedding = embeddings.embed_query(content)

        # Prepare the document to insert into MongoDB
        document = {
            "url": url,
            "content": content,
            "embedding": embedding
        }

        # Save to MongoDB
        save_to_mongodb(document)

        # Print status every 10 articles
        if (index + 1) % 10 == 0:
            print(f"{index + 1} articles processed and stored in MongoDB.")

    print("All articles processed and stored in MongoDB.")

if __name__ == "__main__":
    # Path to the CSV file
    csv_file = "articles.csv"  # Replace with the path to your CSV file

    # Process the CSV and store embeddings in MongoDB
    process_csv_and_store_embeddings(csv_file)