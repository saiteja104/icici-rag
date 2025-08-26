import os
from sentence_transformers import SentenceTransformer
from chunking import load_and_chunk_loan_data, save_chunks_locally, evaluate_chunking_quality
from pinecone_utils import initialize_pinecone
from embedding import create_and_store_embeddings
from dotenv import load_dotenv
load_dotenv()

def main():
    print("Starting ICICI Loan Data Processing Pipeline...")

    #Config
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
    print(PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME)
    # RAW_DATA_FILE = r"D:\icici_rag\data_crawling\data\home_loan_raw_data.json"
    RAW_DATA_FILE = r"D:\icici_rag\data\loan_raw_data_final.json"

    if not PINECONE_API_KEY:
        raise ValueError("Missing Pinecone API key. Set PINECONE_API_KEY as environment variable.")

    try:
        # Step 1: Load + chunk
        chunks = load_and_chunk_loan_data(RAW_DATA_FILE)

        # Step 2: Evaluate quality
        evaluate_chunking_quality(chunks)

        # Step 3: Save backup
        save_chunks_locally(chunks)

        # Step 4: Embedding model
        print("Loading embedding model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Embedding model loaded")

        # Step 5: Pinecone init
        index = initialize_pinecone(PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME)

        # Step 6: Store embeddings
        create_and_store_embeddings(chunks, index, model)

        # Step 7: Verify storage
        stats = index.describe_index_stats()
        print("\nPipeline Complete!")
        print(f"   Total vectors in Pinecone: {stats.total_vector_count}")
        print(f"   Processed chunks: {len(chunks)}")

    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()