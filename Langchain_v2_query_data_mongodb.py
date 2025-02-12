import argparse
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
import numpy as np
import openai

load_dotenv()

# MongoDB connection details
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "DrugWise"
COLLECTION_NAME = "document_embeddings"

openai.api_key = os.environ['OPENAI_API_KEY']

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():        
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the embedding function
    embedding_function = OpenAIEmbeddings()

    # Generate embedding for the query
    query_embedding = embedding_function.embed_query(query_text)

    # Search the MongoDB database
    results = query_mongodb(query_embedding, top_k=5)
    if len(results) == 0 or results[0][1] < 0.5:
        print(f"\n\nUnable to find matching results.")
        return

    # Prepare the context for the prompt
    context_text = "\n\n---\n\n".join([doc["content"] for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    # Generate response using ChatOpenAI
    model = ChatOpenAI()
    response_text = model.predict(prompt)

    print(f"Response: {response_text}")


def query_mongodb(query_embedding, top_k=5):
    """
    Query MongoDB for the most similar documents based on the query embedding.
    """
    # Connect to MongoDB
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    # Print the first few documents in the collection
    for doc in collection.find().limit(5):
        print("Content:", doc["content"][:200])  # Print the first 200 characters of the content
        print("Embedding:", doc["embedding"][:5])  # Print the first 5 values of the embedding
        print("Metadata:", doc.get("metadata", {}))
        print("-" * 50)

    # Fetch all documents from MongoDB
    documents = list(collection.find({}))

    # Compute cosine similarity between query embedding and document embeddings
    similarities = []
    for doc in documents:
        doc_embedding = np.array(doc["embedding"])
        similarity = np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )
        similarities.append((doc, similarity))

    # Sort by similarity and return top_k results
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


if __name__ == "__main__":
    main()