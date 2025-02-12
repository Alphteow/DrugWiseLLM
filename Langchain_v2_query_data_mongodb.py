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
You are an expert assistant specializing in drug-drug interactions (DDIs). Your role is to provide accurate, concise, and clinically relevant answers to user queries based on the provided context. You must only use the information retrieved from the context to answer the question and avoid adding any external knowledge or assumptions. Your answers should be clear, actionable, and focused on addressing the user's query.

Answer the question based only on the following context. Incorporate the respective full URLs naturally into your response:

{context}

---

Answer the question based on the above context returning the respective full url: {question}
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

    for i, (doc, score) in enumerate(results):
        print(f"{i+1}. Document ID: {doc['url']}, Similarity Score: {score:.4f}")

    # Prepare the context for the prompt
    context_text = "\n\n---\n\n".join(
    [f"{doc['content']} (Full URL: {doc['url']})" for doc, _score in results]
    )
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    # Generate response using ChatOpenAI
    model = ChatOpenAI()
    response_text = model.invoke(prompt)

    print(f"Response: {response_text.content}")


def query_mongodb(query_embedding, top_k=5):
    """
    Query MongoDB for the most similar documents based on the query embedding.
    """
    # Connect to MongoDB
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

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