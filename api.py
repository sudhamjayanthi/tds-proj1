import os
import json
import base64
import numpy as np
import faiss
import openai
import ollama
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Embedding provider can be switched via environment variable
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama")
# LLM provider for final answer generation
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

OLLAMA_EMBED_MODEL = "nomic-embed-text"
OPENAI_EMBED_MODEL = "text-embedding-3-small"
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.1")
OPENAI_LLM_MODEL = "gpt-4o-mini"

# File paths
FAISS_INDEX_PATH = "faiss_index.idx"
METADATA_PATH = "metadata.json"
# Search parameters
TOP_K = 5  # Number of relevant chunks to retrieve

# --- Initialize Application and Load Data ---
app = FastAPI()

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load FAISS index and metadata
if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):
    raise RuntimeError("FAISS index or metadata not found. Please run embed.py first.")

print("Loading FAISS index and metadata...")
index = faiss.read_index(FAISS_INDEX_PATH)
with open(METADATA_PATH, "r") as f:
    metadata_file_content = json.load(f)
# Extract data and configuration from the metadata file
metadata = metadata_file_content["chunks"]
INDEX_EMBEDDING_PROVIDER = metadata_file_content["embedding_provider"]
INDEX_EMBEDDING_MODEL = metadata_file_content["embedding_model"]
print(
    f"Data loaded successfully. Index was created with {INDEX_EMBEDDING_PROVIDER} using model {INDEX_EMBEDDING_MODEL}."
)

# Initialize OpenAI client if needed
openai_client = None
if LLM_PROVIDER == "openai" or INDEX_EMBEDDING_PROVIDER == "openai":
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set in the .env file but is required.")
    openai_client = openai.OpenAI(api_key=openai_api_key)


# --- Helper Functions ---
def get_embedding(text: str):
    """Generates an embedding for the given text using the model the index was built with."""
    if INDEX_EMBEDDING_PROVIDER == "ollama":
        response = ollama.embeddings(model=INDEX_EMBEDDING_MODEL, prompt=text)
        return response["embedding"]
    else:  # openai
        response = openai_client.embeddings.create(
            input=[text], model=INDEX_EMBEDDING_MODEL
        )
        return response.data[0].embedding


def get_llm_response(prompt: str):
    """Gets a response from the LLM using the configured provider."""
    if LLM_PROVIDER == "ollama":
        response = ollama.chat(
            model=OLLAMA_LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful teaching assistant for the 'Tools in Data Science' course.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response["message"]["content"]
    else:  # openai
        completion = openai_client.chat.completions.create(
            model=OPENAI_LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful teaching assistant for the 'Tools in Data Science' course.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return completion.choices[0].message.content


def get_image_description(base64_image: str):
    """Gets a description of the image using the configured LLM provider."""
    # Use the general LLM provider for images as well, assuming it's a vision model
    if LLM_PROVIDER == "ollama":
        try:
            # Note: The model used here should have vision capabilities.
            # You might need a specific vision model from Ollama.
            response = ollama.chat(
                model=os.getenv(
                    "OLLAMA_VLM_MODEL", "llama3.1"
                ),  # Defaulting to the main LLM model
                messages=[
                    {
                        "role": "user",
                        "content": "Describe this image concisely.",
                        "images": [base64_image],
                    }
                ],
            )
            return response["message"]["content"]
        except Exception as e:
            print(f"Error getting image description with Ollama: {e}")
            return "Could not analyze image."
    else:  # openai
        if not openai_client:
            return "Could not analyze image - OpenAI client not configured."
        try:
            response = openai_client.chat.completions.create(
                model=OPENAI_LLM_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image concisely."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/webp;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=100,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error getting image description with OpenAI: {e}")
            return "Could not analyze image."


# --- API Request and Response Models ---
class ApiRequest(BaseModel):
    question: str
    image: Optional[str] = None  # Base64 encoded image


class Link(BaseModel):
    url: str
    text: str


class ApiResponse(BaseModel):
    answer: str
    links: List[Link]


# --- API Endpoint ---
@app.post("/api/", response_model=ApiResponse)
async def process_question(request: ApiRequest):
    """
    Processes a student's question, performs a RAG search, and returns an answer.
    """
    question = request.question

    # If an image is attached, get a description and prepend it to the question
    if request.image:
        print("Image detected, generating description...")
        image_desc = get_image_description(request.image)
        question = f"An image is attached with the following description: '{image_desc}'. Question: {question}"
        print(f"Modified question with image context: {question}")

    # 1. Embed the user's question and Search for context
    retrieved_chunks = []
    try:
        print("Embedding user question...")
        question_embedding = np.array([get_embedding(question)]).astype("float32")

        print(f"Searching for top {TOP_K} relevant chunks...")
        distances, indices = index.search(question_embedding, TOP_K)
        retrieved_chunks = [metadata[i] for i in indices[0]]

    except AssertionError as e:
        # This is a critical error, likely a dimension mismatch between query and index.
        query_dim = (
            question_embedding.shape[1]
            if "question_embedding" in locals()
            else "unknown"
        )
        print("=" * 50)
        print(f"!!! FAISS ASSERTION ERROR !!!")
        print(f"DETAILS: {e}")
        print(
            f"This usually means the embedding model used for the query is different from the model used to build the index."
        )
        print(f"Index Dimension (index.d): {index.d}")
        print(f"Query Vector Dimension: {query_dim}")
        print("Falling back to LLM without RAG context.")
        print("=" * 50)

    except Exception as e:
        # Catch any other errors during embedding or search
        print("=" * 50)
        print(f"!!! UNEXPECTED ERROR during embedding/search: {e} !!!")
        print("Falling back to LLM without RAG context.")
        print("=" * 50)

    # 3. Retrieve context and format links
    context = (
        "\n\n---\n\n".join([chunk["text"] for chunk in retrieved_chunks])
        if retrieved_chunks
        else "No relevant context found."
    )

    # Create unique links from the retrieved chunks
    unique_links = {}
    for chunk in retrieved_chunks:
        if chunk["url"] not in unique_links:
            link_text = chunk.get("title", chunk.get("source_file", "Discussion Post"))
            unique_links[chunk["url"]] = Link(url=chunk["url"], text=link_text)

    # 4. Construct the prompt for the LLM
    prompt = f"""You are a helpful Virtual Teaching Assistant for the 'Tools in Data Science' course.
Your goal is to answer a student's question based on the provided context from the course's Discourse forum.
The answer must be concise, accurate, and directly address the question.
Do not make up information. If the context does not provide an answer, state that you couldn't find a relevant answer in the provided materials.

CONTEXT FROM DISCOURSE:
---
{context}
---

STUDENT'S QUESTION:
{question}

Based on the context, provide your answer."""

    # 5. Get the final answer from the LLM
    print(f"Generating final answer with {LLM_PROVIDER}...")
    try:
        answer = get_llm_response(prompt)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get response from LLM: {e}"
        )

    # 6. Format and return the response
    return ApiResponse(answer=answer.strip(), links=list(unique_links.values()))


# --- Root endpoint for health check ---
@app.get("/")
def read_root():
    return {
        "status": "Virtual TA API is running",
        "providers": {"embedding": INDEX_EMBEDDING_PROVIDER, "llm": LLM_PROVIDER},
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
