import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from dotenv import load_dotenv
import openai
from langchain.memory import ConversationBufferMemory
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API Key
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OpenAI API Key not found. Please add it to the .env file.")
openai.api_key = API_KEY

# Fine-tuned model name
FINE_TUNED_MODEL = "ft:gpt-4o-mini-2024-07-18:personal::AabQd3B8"

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def load_json_files(directory: str) -> List[Dict]:
    print(f"Loading JSON files from directory: {directory}")
    files = Path(directory).glob("*.json")
    cases = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            case = json.load(f)
            for opinion in case.get("casebody", {}).get("opinions", []):
                enriched_case = {
                    "id": case.get("id", ""),
                    "case_name": case.get("name", ""),
                    "text": opinion.get("text", "").strip(),
                    "decision_date": case.get("decision_date", ""),
                    "jurisdiction": case.get("jurisdiction", {}).get("name", "Unknown"),
                }
                cases.append(enriched_case)
    print(f"Loaded {len(cases)} cases successfully.")
    return cases

def load_va_constitution(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    sections = text.split('\n\n')
    va_sections = []
    for idx, section in enumerate(sections):
        if section.strip():
            va_sections.append({
                'id': f'va_constitution_{idx}',
                'text': section.strip(),
                'case_name': f'VA Constitution Section {idx}',
            })
    return va_sections

def create_embeddings(data: List[Dict], model_name="all-MiniLM-L6-v2", pkl_file="finaldata1.pkl") -> None:
    print(f"Generating embeddings for {len(data)} cases...")
    model = SentenceTransformer(model_name)
    embeddings = []
    for entry in data:
        try:
            embedding = model.encode(entry["text"], normalize_embeddings=True)
            embeddings.append({"embedding": embedding, "metadata": entry})
        except Exception as e:
            print(f"Error processing entry {entry.get('id')}: {e}")

    with open(pkl_file, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"Embeddings saved to {pkl_file}.")

def load_faiss_index(pkl_file="finaldata1.pkl") -> Tuple[faiss.IndexFlatIP, List[Dict]]:
    print(f"Loading FAISS index from {pkl_file}...")
    with open(pkl_file, "rb") as f:
        combined_data = pickle.load(f)

    embeddings = np.vstack([entry["embedding"] for entry in combined_data])
    metadata = [entry["metadata"] for entry in combined_data]

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    print(f"FAISS index created with {len(metadata)} items.")
    return index, metadata

def extract_named_entities(text: str) -> List[str]:
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    chunks = ne_chunk(pos_tags, binary=False)
    named_entities = []
    for chunk in chunks:
        if isinstance(chunk, Tree):
            named_entity = " ".join(c[0] for c in chunk)
            named_entities.append(named_entity)
    return named_entities

