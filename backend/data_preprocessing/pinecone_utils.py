# pinecone_utils.py
from pinecone import Pinecone, ServerlessSpec

def initialize_pinecone(api_key: str, region: str, index_name: str = "icici-home-loans"):
    """Initialize Pinecone connection and index (v3 SDK)"""
    pc = Pinecone(api_key=api_key)

    dimension = 384  # all-MiniLM-L6-v2 model

    # Get existing indexes
    existing_indexes = [i["name"] for i in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        print(f"Creating new Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=region),  # e.g. "us-east-1"
        )
        print("Waiting for index to be ready...")

    index = pc.Index(index_name)
    print(f"Pinecone index '{index_name}' ready")
    return index