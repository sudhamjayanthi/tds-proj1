import os
import glob
import json
import numpy as np
import faiss
from semantic_text_splitter import MarkdownSplitter
from tokenizers import Tokenizer
import ollama
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Choose your embedding provider: "ollama" or "openai"
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "ollama")
# Models
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OPENAI_EMBED_MODEL = "text-embedding-3-small"
# Max tokens per chunk for the splitter
MAX_TOKENS = 512
# Directories containing the markdown files
DISCOURSE_CONTENT_DIR = "discourse_content"
COURSE_CONTENT_DIR = "course_content"
# Output files
FAISS_INDEX_PATH = "faiss_index.idx"
METADATA_PATH = "metadata.json"

# --- Initialize Clients ---
# Initialize OpenAI client if using it
if EMBEDDING_PROVIDER == "openai":
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set in the .env file.")
    openai.api_key = openai_api_key


def get_embedding(text, provider=EMBEDDING_PROVIDER):
    """
    Generates an embedding for the given text using the configured provider.
    """
    if provider == "ollama":
        response = ollama.embeddings(model=OLLAMA_EMBED_MODEL, prompt=text)
        return response["embedding"]
    elif provider == "openai":
        response = openai.embeddings.create(input=[text], model=OPENAI_EMBED_MODEL)
        return response.data[0].embedding
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


def process_discourse_files():
    """Process discourse content files with format: id-slug.md"""
    all_chunks = []
    md_files = glob.glob(os.path.join(DISCOURSE_CONTENT_DIR, "*.md"))

    print(f"Found {len(md_files)} discourse files to process.")

    for filepath in md_files:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        filename = os.path.splitext(os.path.basename(filepath))[0]

        # Parse filename: id-slug format
        if "-" in filename:
            parts = filename.split("-", 1)  # Split on first dash only
            topic_id = parts[0]
            topic_slug = parts[1]
            url = f"https://discourse.onlinedegree.iitm.ac.in/t/{topic_slug}/{topic_id}"
            title = f"Discussion: {topic_slug.replace('-', ' ').title()}"
        else:
            # Fallback for old format
            topic_id = filename
            url = f"https://discourse.onlinedegree.iitm.ac.in/t/{topic_id}"
            title = f"Discussion Post {topic_id}"

        # Split into chunks
        tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
        splitter = MarkdownSplitter.from_huggingface_tokenizer(tokenizer, MAX_TOKENS)
        chunks = splitter.chunks(content)

        for i, chunk_text in enumerate(chunks):
            all_chunks.append(
                {
                    "text": chunk_text,
                    "url": url,
                    "title": title,
                    "chunk_index": i,
                    "source_file": os.path.basename(filepath),
                    "content_type": "discourse",
                }
            )

    return all_chunks


def process_course_files():
    """Process course content files with format: filename.md"""
    all_chunks = []
    md_files = glob.glob(os.path.join(COURSE_CONTENT_DIR, "*.md"))

    print(f"Found {len(md_files)} course files to process.")

    for filepath in md_files:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        filename = os.path.splitext(os.path.basename(filepath))[0]
        url = f"https://tds.s-anand.net/#/{filename}"

        # Extract title from first heading or use filename
        first_line = content.split("\n")[0] if content else ""
        if first_line.startswith("#"):
            title = first_line.replace("#", "").strip()
        else:
            title = filename.replace("-", " ").title()

        # Split into chunks
        tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
        splitter = MarkdownSplitter.from_huggingface_tokenizer(tokenizer, MAX_TOKENS)
        chunks = splitter.chunks(content)

        for i, chunk_text in enumerate(chunks):
            all_chunks.append(
                {
                    "text": chunk_text,
                    "url": url,
                    "title": title,
                    "chunk_index": i,
                    "source_file": os.path.basename(filepath),
                    "content_type": "course",
                }
            )

    return all_chunks


def prepare_and_save_data():
    """
    Loads markdown files from both directories, splits them into chunks, creates embeddings,
    and saves them to a FAISS index and a metadata file.
    """
    print("Starting data preparation...")

    # Process both content types
    discourse_chunks = process_discourse_files()
    course_chunks = process_course_files()

    all_chunks_with_meta = discourse_chunks + course_chunks
    print(
        f"Created {len(all_chunks_with_meta)} total chunks ({len(discourse_chunks)} discourse + {len(course_chunks)} course)."
    )

    if not all_chunks_with_meta:
        print("No chunks found. Aborting.")
        return

    # 2. Generate Embeddings
    print(f"Generating embeddings using {EMBEDDING_PROVIDER}...")
    all_embeddings = []
    for i, item in enumerate(all_chunks_with_meta):
        all_embeddings.append(get_embedding(item["text"]))
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(all_chunks_with_meta)} chunks")

    embeddings_np = np.array(all_embeddings).astype("float32")

    if embeddings_np.size == 0:
        print("No embeddings were generated. Cannot create FAISS index.")
        return

    dimension = embeddings_np.shape[1]
    print(f"Embeddings generated with dimension: {dimension}")

    # 3. Create and Save FAISS Index
    print("Creating FAISS index...")
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"FAISS index saved to {FAISS_INDEX_PATH}")

    # 4. Save Metadata
    output_metadata = {
        "embedding_provider": EMBEDDING_PROVIDER,
        "embedding_model": OLLAMA_EMBED_MODEL
        if EMBEDDING_PROVIDER == "ollama"
        else OPENAI_EMBED_MODEL,
        "chunks": all_chunks_with_meta,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(output_metadata, f, indent=4)
    print(f"Metadata saved to {METADATA_PATH}")

    print("Data preparation complete.")


if __name__ == "__main__":
    prepare_and_save_data()
