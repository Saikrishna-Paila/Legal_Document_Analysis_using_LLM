import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from langchain.memory import ConversationBufferMemory
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API Key
API_KEY = os.getenv("OPENAI_API_KEY") # create .env file keep you openAI key (OPENAI_API_KEY= your openAI API in the .env)
if not API_KEY:
    raise ValueError("OpenAI API Key not found. Please add it to the .env file.")
openai.api_key = API_KEY

# Fine-tuned model name
FINE_TUNED_MODEL = "use your open AI fine tuned model"

# Initialize memory for conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize tracking variables
total_queries = 0
evasion_count = 0
successfully_answered = 0


def load_json_files(directory: str) -> List[Dict]:
    """
    Load JSON files from a directory and prepare metadata for embeddings.
    """
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


def create_embeddings(data: List[Dict], model_name="all-MiniLM-L6-v2", pkl_file="finaldata1.pkl") -> None:
    """
    Generate embeddings for RAG data and save them to a pickle file.
    """
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
    """
    Load embeddings and metadata from the pickle file and create a FAISS index.
    """
    print(f"Loading FAISS index from {pkl_file}...")
    with open(pkl_file, "rb") as f:
        combined_data = pickle.load(f)

    embeddings = np.vstack([entry["embedding"] for entry in combined_data])
    metadata = [entry["metadata"] for entry in combined_data]

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    print(f"FAISS index created with {len(metadata)} items.")
    return index, metadata


def query_rag(query: str, index, metadata) -> Tuple[str, bool]:
    """
    Query the RAG system and return the response along with a relevance flag.
    """
    print(f"Querying RAG system for: {query}")
    query_embedding = SentenceTransformer("all-MiniLM-L6-v2").encode([query])[0]
    distances, indices = index.search(np.array([query_embedding]), 5)

    relevant_docs = [
        metadata[idx] for idx, distance in zip(indices[0], distances[0]) if distance > 0.1
    ]

    if not relevant_docs:
        print("No relevant information found in RAG.")
        return "No relevant information found.", False

    context = "\n\n".join([
        f"{doc.get('case_name', 'Document')}:\n{doc.get('text', '')[:1500]}"
        for doc in relevant_docs
    ])
    print("Relevant information retrieved from RAG.")
    return context, True


def query_fine_tuned_model(prompt: str) -> str:
    """
    Query the fine-tuned model using the updated OpenAI API.
    """
    print("Querying fine-tuned model...")
    try:
        response = openai.ChatCompletion.create(
            model=FINE_TUNED_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error querying fine-tuned model: {e}")
        return f"Error querying fine-tuned model: {e}"


def log_successful_query(query: str, context: str, response: str, log_file="successful_queries.json"):
    """
    Log successful queries and their contexts.
    """
    log_entry = {
        "query": query,
        "context": context,
        "response": response
    }

    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            existing_logs = json.load(f)
    else:
        existing_logs = []

    existing_logs.append(log_entry)

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(existing_logs, f, ensure_ascii=False, indent=4)
    print(f"Logged successful query to {log_file}.")


def build_prompt(rag_context: str, query: str, learning_examples: List[Dict], num_examples: int = 3) -> str:
    """
    Build a prompt by combining RAG context, query, and a few learning examples.
    """
    example_texts = "\n\n".join([
        f"Example {i+1}:\nCase Name: {example['case_name']}\nDecision Date: {example['decision_date']}\nSummary: {example['opinion_summary']}"
        for i, example in enumerate(learning_examples[:num_examples])
    ])

    prompt = f"""
    You are a legal assistant (name: AskLaw) tasked with answering questions based on the following context and examples.

    Context:
    {rag_context}

    Learning Examples:
    {example_texts}
    Note: Ensure to provide detailed information about the case ID when mentioned.


    Question:
    {query}

    Answer:
    """
    return prompt


def hybrid_query_system(query: str, index, metadata, learning_examples: List[Dict], log_file="successful_queries.json") -> str:
    """
    Query both the RAG system and fine-tuned model to generate the best response using learning examples.
    """
    global successfully_answered, evasion_count

    # Query the RAG system
    rag_context, rag_relevant = query_rag(query, index, metadata)

    # Build the prompt
    prompt = build_prompt(rag_context, query, learning_examples, num_examples=3)

    # Query fine-tuned model with the combined prompt
    fine_tuned_response = query_fine_tuned_model(prompt)

    if "No relevant information found" not in rag_context:
        successfully_answered += 1
        log_successful_query(query, rag_context, fine_tuned_response, log_file)
    else:
        evasion_count += 1

    return fine_tuned_response


def load_learning_examples(file_path: str) -> List[Dict]:
    """
    Load learning examples from a JSON file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_learning_examples(examples: List[Dict], limit: int = 5) -> None:
    """
    Print a specified number of learning examples.
    """
    print("\n--- Learning Examples ---")
    for i, example in enumerate(examples[:limit], start=1):
        print(f"Example {i}:")
        print(f"  Case ID: {example['case_id']}")
        print(f"  Case Name: {example['case_name']}")
        print(f"  Decision Date: {example['decision_date']}")
        print(f"  Opinion Summary: {example['opinion_summary']}\n")


if __name__ == "__main__":
    # Prepare data
    data_directory = "data/"
    cases = load_json_files(data_directory)

    # Load learning examples
    examples_file = "learning_examples.json"  # Replace with actual path to the file
    try:
        learning_examples = load_learning_examples(examples_file)
        print(f"Loaded {len(learning_examples)} learning examples successfully.")
    except FileNotFoundError:
        print(f"Error: The file '{examples_file}' was not found. Please check the path.")
        learning_examples = []

    # Create embeddings and load FAISS index
    pkl_file = "finaldata1.pkl"
    if not os.path.exists(pkl_file):
        create_embeddings(cases, pkl_file=pkl_file)
    index, metadata = load_faiss_index(pkl_file)

    while True:
        user_input = input("Enter your query (type 'exit' to quit, 'examples' to see learning examples): ").strip()
        if user_input.lower() == "exit":
            print("Exiting the program. Goodbye!")
            print(f"\nTotal Queries: {total_queries}")
            print(f"Successfully Answered: {successfully_answered}")
            print(f"Evasion Count: {evasion_count}")
            break
        elif user_input.lower() == "examples":
            if learning_examples:
                print_learning_examples(learning_examples, limit=5)
            else:
                print("No learning examples loaded.")
        else:
            total_queries += 1
            final_response = hybrid_query_system(user_input, index, metadata, learning_examples)
            print(f"\n{final_response}")
