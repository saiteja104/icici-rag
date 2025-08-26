from __future__ import annotations
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from contextlib import asynccontextmanager

# -------- Config --------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_REGION = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "icici-home-loans")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "5"))

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY missing. Put it in your .env file.")

# Globals
index = None
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global index, model
    print("Loading model and connecting to Pinecone...")

    model = SentenceTransformer(EMBED_MODEL_NAME)
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if PINECONE_INDEX_NAME not in [i.name for i in pc.list_indexes()]:
        raise RuntimeError(f"Pinecone index '{PINECONE_INDEX_NAME}' does not exist")

    index = pc.Index(PINECONE_INDEX_NAME)
    print("Pinecone + model ready")

    yield
    print("Shutting down app...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- Models --------
class QueryRequest(BaseModel):
    query: str = Field(..., description="User query text")
    top_k: int = Field(TOP_K_DEFAULT, ge=1, le=50, description="Number of results")
    filter: Optional[Dict[str, Any]] = None

class Match(BaseModel):
    id: str
    score: float
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    query: str
    top_k: int
    matches: List[Match]

# -------- Routes --------
@app.get("/health")
def health():
    return {"status": "ok", "index": PINECONE_INDEX_NAME, "model": EMBED_MODEL_NAME}

@app.post("/query", response_model=QueryResponse)
def query_index(body: QueryRequest):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query text is required")

    try:
        vector = model.encode([body.query], convert_to_tensor=False)[0].tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    try:
        res = index.query(
            vector=vector,
            top_k=body.top_k,
            include_metadata=True,
            filter=body.filter,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone query failed: {e}")

    matches: List[Match] = []
    for m in res.get("matches", []):
        md = m.get("metadata", {}) or {}
        snippet = md.get("content")
        if snippet and len(snippet) > 260:
            md["content_snippet"] = snippet[:260] + "…"
            md.pop("content", None)  # drop large content
        matches.append(Match(id=m.get("id", ""), score=float(m.get("score", 0.0)), metadata=md))

    return QueryResponse(query=body.query, top_k=body.top_k, matches=matches)



import requests

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY missing. Put it in your .env file.")


class RagResponse(BaseModel):
    query: str
    answer: str
    context: List[str]  # the retrieved chunks


@app.post("/rag", response_model=RagResponse)
def rag_answer(body: QueryRequest):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query text is required")

    # Step 1: Retrieve from Pinecone
    try:
        vector = model.encode([body.query], convert_to_tensor=False)[0].tolist()
        res = index.query(
            vector=vector,
            top_k=body.top_k,
            include_metadata=True,
            filter=body.filter,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pinecone query failed: {e}")

    # Extract contexts
    contexts = []
    for m in res.get("matches", []):
        md = m.get("metadata", {}) or {}
        if "content" in md:
            contexts.append(md["content"])

    if not contexts:
        raise HTTPException(status_code=404, detail="No relevant chunks found")

    # Step 2: Call Groq API
    try:
        prompt = (
            "You are an assistant answering questions about ICICI Bank Home Loans.\n\n"
            f"User question: {body.query}\n\n"
            "Here are some relevant context snippets:\n"
            + "\n\n".join(contexts[:5])  # limit to top 5 chunks for efficiency
            + "\n\nBased on this information, give a helpful, concise, and accurate answer."
        )

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": "You are a financial assistant specialized in ICICI Bank Home Loans."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 512,
        }

        groq_res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        groq_res.raise_for_status()
        answer = groq_res.json()["choices"][0]["message"]["content"]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API call failed: {e}")

    return RagResponse(query=body.query, answer=answer, context=contexts)




@app.get("/")
def root():
    return {"message": "Pinecone RAG Search API"}
