# chunking.py
import re
import json
import time
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os


def load_and_chunk_loan_data(json_file: str) -> List[Dict[str, Any]]:
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        print(f"Loaded {len(raw_data)} documents from {json_file}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find {json_file}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in {json_file}")

    # derive loan_type dynamically from filename (strip _chunks.json or .json)
    loan_type = os.path.basename(json_file).replace("_chunks.json", "").replace(".json", "")
    # fixed chunking strategy
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "• ", "- ", " "],
        keep_separator=True
    )

    all_chunks = []
    seen_chunks = set()  # track unique chunks


    for item_idx, item in enumerate(raw_data):
        category = item.get('category')
        content = item.get('content')

        if len(content) < 50:
            continue

        chunks = splitter.split_text(content)

        for chunk_idx, chunk in enumerate(chunks):
            chunk_text = chunk.strip()

            normalized_text = " ".join(chunk_text.lower().split())
            if len(normalized_text) <= 50 or normalized_text in seen_chunks:
                continue
            seen_chunks.add(normalized_text)


            # skip if too short or duplicate
            # if len(chunk_text) <= 50 or chunk_text in seen_chunks:
            #     continue
            # seen_chunks.add(chunk_text)  # remember this chunk

            chunk_obj = {
                'id': f"{loan_type}_{category}_{hash(item.get('url', ''))}_{chunk_idx}",
                'content': chunk_text,
                'metadata': {
                    'loan_type': loan_type,   
                    'category': category,
                    'source_url': item.get('url', ''),
                    'title': item.get('title', ''),
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks),
                    'content_length': len(chunk_text),
                    'source_doc_index': item_idx,
                    'last_updated': item.get('last_scraped', time.strftime('%Y-%m-%d %H:%M:%S'))
                }
            }
            all_chunks.append(chunk_obj)

    print(f"Created {len(all_chunks)} unique chunks from {len(raw_data)} documents")
    return all_chunks

# def save_chunks_locally(chunks: List[Dict[str, Any]], filename: str = 'loan_chunks.json') -> None:
#     """Save processed chunks locally for backup"""
#     try:
#         save_dir = r"D:\icici_rag\data"
#         os.makedirs(save_dir, exist_ok=True)
#         filepath = os.path.join(save_dir, filename)
#         with open(filepath, 'w', encoding='utf-8') as f:
#             json.dump(chunks, f, indent=2, ensure_ascii=False)
#         print(f"Saved {len(chunks)} chunks to {filepath}")
#     except Exception as e:
#         print(f"Error saving chunks locally: {e}")

def save_chunks_locally(chunks: List[Dict[str, Any]], filename: str = 'loan_chunks1.json') -> None:
    """Save processed chunks locally for backup, appending to existing JSON array."""
    try:
        save_dir = r"D:\icici_rag\data"
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, filename)

        # Load existing data if file exists
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                try:
                    existing_data = json.load(f)
                    if not isinstance(existing_data, list):
                        existing_data = []
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []

        # Append new chunks
        existing_data.extend(chunks)

        # Save back to file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(chunks)} new chunks (total now {len(existing_data)}) to {filepath}")

    except Exception as e:
        print(f"Error saving chunks locally: {e}")

        
def evaluate_chunking_quality(chunks: List[Dict[str, Any]]) -> None:
    """Evaluate and display chunk quality metrics"""
    if not chunks:
        print("No chunks to evaluate.")
        return

    chunk_sizes = [len(chunk['content']) for chunk in chunks]
    categories = {}

    for chunk in chunks:
        cat = chunk['metadata']['category']
        categories[cat] = categories.get(cat, 0) + 1

    print("\nChunking Quality Report:")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Average chunk size: {sum(chunk_sizes) / len(chunk_sizes):.0f} characters")
    print(f"  Chunk size range: {min(chunk_sizes)} - {max(chunk_sizes)} characters")

    print("\n  Chunks per category:")
    for cat, count in sorted(categories.items()):
        avg_size = sum(len(c['content']) for c in chunks if c['metadata']['category'] == cat) / count
        print(f"    {cat}: {count} chunks (avg: {avg_size:.0f} chars)")