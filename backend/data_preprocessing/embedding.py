# embedding.py
import time
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone


def create_and_store_embeddings(
    chunks: List[Dict[str, Any]],
    index,  # index is a Pinecone.Index object, but v3 SDK has no type export
    model: SentenceTransformer,
    batch_size: int = 100
) -> None:
    """Create embeddings and store in Pinecone"""
    total_batches = (len(chunks) - 1) // batch_size + 1
    print(f"Processing {len(chunks)} chunks in {total_batches} batches...")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = i // batch_size + 1

        print(f"Processing batch {batch_num}/{total_batches}...")

        try:
            texts = [chunk['content'] for chunk in batch]
            embeddings = model.encode(texts, convert_to_tensor=False, show_progress_bar=False)

            vectors = []
            for j, chunk in enumerate(batch):
                metadata = chunk['metadata'].copy()
                metadata['content'] = chunk['content'][:1000]  # Pinecone metadata limit

                vectors.append((
                    chunk['id'],
                    embeddings[j].tolist(),
                    metadata
                ))

            # Upsert into Pinecone (v3)
            index.upsert(vectors=vectors)
            print(f" Uploaded batch {batch_num}/{total_batches}")
            time.sleep(0.5)

        except Exception as e:
            print(f"Error in batch {batch_num}: {e}")
            continue

    print(f"Successfully stored {len(chunks)} embeddings in Pinecone")
