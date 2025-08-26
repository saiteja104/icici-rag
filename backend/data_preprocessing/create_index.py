from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
load_dotenv()
import os
# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Index name
# index_name = "icici-index"
index_name = "icici-loans-index"
# Check if index already exists
if index_name not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,  # must match your embedding model dimension
        metric="cosine",  # similarity metric
        spec=ServerlessSpec(
            cloud="aws",  # or "gcp"
            region="us-east-1"
        )
    )
    print(f"Created index: {index_name}")
else:
    print(f"Index '{index_name}' already exists.")
