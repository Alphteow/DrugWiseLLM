from pymongo import MongoClient

# MongoDB connection details
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "DrugWise"
COLLECTION_NAME = "document_embeddings"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Count the number of documents in the collection
doc_count = collection.count_documents({})
print(f"Number of documents in the collection: {doc_count}")

# Print one document to verify
if doc_count > 0:
    print("Sample document:", collection.find_one())
else:
    print("The collection is empty.")