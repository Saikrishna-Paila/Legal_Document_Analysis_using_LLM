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
def summarize_text(text, max_tokens=500):
    """
    Summarize a given text to fit within token constraints while retaining key arguments.
    """
    llm = ChatOpenAI(model_name="gpt-4", max_tokens=max_tokens, temperature=0)
    prompt = f"Summarize the following legal arguments concisely:\n\n{text}"
    return llm.predict(prompt).strip()


def query_rag(query: str, index, metadata) -> Tuple[str, bool]:
    query_embedding = SentenceTransformer("all-MiniLM-L6-v2").encode([query])[0]
    distances, indices = index.search(np.array([query_embedding]), 5)

    relevant_docs = [
        metadata[idx] for idx, dist in zip(indices[0], distances[0]) if dist > 0.1
    ]

    if not relevant_docs:
        return "No relevant information found.", False

    context = "\n\n".join([
        f"{doc.get('case_name', 'Document')}:\n{doc.get('text', '')[:1500]}"
        for doc in relevant_docs
    ])
    return context, True

def log_successful_query(query: str, context: str, response: str, log_file="successful_queries.json"):
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

def build_prompt(rag_context: str, query: str) -> str:
    prompt = f"""
    You are a legal assistant tasked with answering questions based on the following context.

    Context:
    {rag_context}

    Question:
    {query}

    Answer:
    """
    return prompt

def query_fine_tuned_model(prompt: str) -> str:
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

def hybrid_query_system(query: str, index, metadata, fine_tuned_model=False) -> str:
    rag_context, rag_relevant = query_rag(query, index, metadata)
    if not rag_relevant:
        return "No relevant information found in the knowledge base."

    if fine_tuned_model:
        prompt = build_prompt(rag_context, query)
        response = query_fine_tuned_model(prompt)
        log_successful_query(query, rag_context, response)
        return response

    return rag_context
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
def load_learning_examples(file_path: str) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
def create_rag_pipeline(index, metadata):
    """
    Create a RAG pipeline with FAISS, LangChain, and enhanced memory tracking.
    """
    # Update to the latest ChatOpenAI class
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.3)
    # Modify memory to store only the last interaction
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    context_store = {"last_query": "", "last_answer": ""}

    def filter_retrieved_docs(query: str, top_k=5) -> List[Dict]:
        query_embedding = SentenceTransformer("all-MiniLM-L6-v2").encode([query])[0]
        distances, indices = index.search(np.array([query_embedding]), top_k)

        relevant_docs = []
        for i, idx in enumerate(indices[0]):
            if distances[0][i] > 0.1:  # Lowered the threshold to include more documents
                doc = metadata[idx]
                relevant_docs.append(doc)

        return relevant_docs

    def query_rag(query: str, context_needed=True) -> str:
        # Retrieve relevant documents from the index
        relevant_docs = filter_retrieved_docs(query, top_k=10)  # Increased top_k

        if not relevant_docs:
            return "No relevant information found in the knowledge base."

        # Limit the length of each document
        max_chars_per_doc = 1500  # Increased to allow more content
        context = "\n\n".join([
            f"{doc.get('case_name', doc.get('id', 'Document'))}:\n{doc.get('text', 'No text available')[:max_chars_per_doc]}"
            for doc in relevant_docs
        ])

        # Reintroduce few-shot examples
        few_shot_examples = """
Example 1:
Context:
VA Constitution Section 1:
"That all men are by nature equally free and independent and have certain inherent rights..."

Case:
John Doe alleges that his right to free speech has been violated by a new state law restricting public demonstrations.

Based on the above context, provide the final verdict for the case, being concise and focusing only on the verdict.

Response:
The state law restricting public demonstrations violates John Doe's inherent right to free speech as protected by the VA Constitution. Therefore, the law is unconstitutional, and John Doe's right has been violated.

Example 2:
Context:
Commonwealth vs. Jane Smith:
"Jane Smith was charged with grand larceny after allegedly stealing goods valued over $500."

Case:
List cases involving grand larceny.

Response:
Cases involving grand larceny include Commonwealth vs. Jane Smith and Commonwealth vs. John Doe.

"""

        # Construct the prompt
        prompt = f"""
You are a legal assistant (name: AskLAW) tasked with answering questions based on the following context from the Virginia Constitution and relevant case law. If the context does not provide the answer, use your own knowledge to answer the question.

{few_shot_examples}

Now, please answer the following question:

Context:
{context}

Question:
{query}

Answer:
"""

        # Ensure the prompt is within the token limit
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model("gpt-4")
            token_count = len(encoding.encode(prompt))
            max_tokens_allowed = 8192  # GPT-4's maximum context length

            if token_count > max_tokens_allowed:
                # Adjust max_chars_per_doc to reduce token count
                reduction_ratio = max_tokens_allowed / token_count
                new_max_chars_per_doc = int(max_chars_per_doc * reduction_ratio * 0.9)  # Slightly less to be safe

                # Reconstruct the context with the new max chars per doc
                context = "\n\n".join([
                    f"{doc.get('case_name', doc.get('id', 'Document'))}:\n{doc.get('text', 'No text available')[:new_max_chars_per_doc]}"
                    for doc in relevant_docs
                ])

                # Reconstruct the prompt
                prompt = f"""
You are a legal assistant (name: AskLAW) tasked with answering questions based on the following context from the Virginia Constitution and relevant case law. If the context does not provide the answer, use your own knowledge to answer the question.

{few_shot_examples}

Now, please answer the following question:

Context:
{context}

Question:
{query}

Answer:
"""
                # Recalculate token count
                token_count = len(encoding.encode(prompt))
                if token_count > max_tokens_allowed:
                    return "The context is too large to process. Please simplify your query."
        except ImportError:
            print("tiktoken library is not installed. Install it using 'pip install tiktoken'")
            return "Token counting failed due to missing library."

        response = llm.invoke(prompt).content.strip()

        # Update context store
        context_store["last_query"] = query
        context_store["last_answer"] = response

        # Optionally, manage conversation history
        memory.clear()
        memory.save_context({"input": query}, {"output": response})

        return response

    return query_rag, memory, context_store



def run_rag_pipeline(uploaded_file: str = None, user_query: str = None) -> str:
    """
    Refactored RAG pipeline function for Streamlit integration.

    Args:
        uploaded_file (str): Path to the uploaded JSON file for processing.
        user_query (str): User-provided query for processing.

    Returns:
        str: The response generated based on the input mode.
    """
    directory = "Case 1 JSON"
    pkl_file = "finaldata1.pkl"

    # Load case data
    cases = load_json_files(directory)

    # Load VA Constitution file (path is fixed)
    va_constitution_path = "va_constitution.json"
    va_sections = load_va_constitution(va_constitution_path)

    # Combine cases and VA Constitution sections
    all_data = cases + va_sections

    # Create embeddings and save to pickle file if not already done
    if not os.path.exists(pkl_file):
        create_embeddings(all_data, pkl_file=pkl_file)

    # Load FAISS index and metadata
    index, metadata = load_faiss_index(pkl_file)
    query_rag, memory, context_store = create_rag_pipeline(index, metadata)

    # If a JSON file is uploaded
    if uploaded_file:
        try:
            with open(uploaded_file, "r", encoding="utf-8") as f:
                test_file_data = json.load(f)

            # Combine text of all opinions into one query
            combined_query = " ".join([
                opinion.get("text", "").strip()
                for opinion in test_file_data.get("casebody", {}).get("opinions", [])
                if opinion.get("text")
            ])

            # Fallback to head_matter if opinions are empty
            if not combined_query.strip():
                combined_query = test_file_data["casebody"].get("head_matter", "").strip()

            # Add a default question if necessary
            if not combined_query.strip():
                combined_query = (
                    "This case involves legal arguments but lacks detailed opinion text. "
                    "Based on the available context, what is the expected verdict?"
                )

            # Summarize if needed
            if len(combined_query) > 2000:  # Adjust threshold as needed
                combined_query = summarize_text(combined_query)

            # Get the expected verdict for the test file
            response = query_rag(combined_query)
            return f"Expected Verdict for Test File:\n{response}"

        except Exception as e:
            return f"Error processing uploaded file: {str(e)}"

    # If a query is provided
    elif user_query:
        try:
            response = query_rag(user_query)
            return f"Answer:\n{response}"
        except Exception as e:
            return f"Error processing query: {str(e)}"

    # If neither input is provided
    return "Please provide a JSON file or a query for processing."






def run_fine_tuned_model(user_input: str, pkl_file="finaldata1.pkl", data_directory="Case 1 JSON/") -> str:
    """
    Processes a user query dynamically, querying the fine-tuned model.

    Args:
        user_input (str): The user's query.
        pkl_file (str): Path to the embeddings file for FAISS.
        data_directory (str): Directory containing JSON case files.

    Returns:
        str: The response based on the user query.
    """
    # Load cases
    cases = load_json_files(data_directory)

    # Create embeddings and load FAISS index
    if not os.path.exists(pkl_file):
        create_embeddings(cases, pkl_file=pkl_file)
    index, metadata = load_faiss_index(pkl_file)

    # Process query using hybrid query system with fine-tuned model
    final_response = hybrid_query_system(user_input, index, metadata, fine_tuned_model=True)
    return final_response



if __name__ == "__main__":
    print("Welcome! Choose your mode of operation:")
    print("1. RAG Pipeline")
    print("2. Fine-Tuned Model")
    mode_selection = input("Enter 1 for RAG Pipeline or 2 for Fine-Tuned Model: ").strip()

    if mode_selection == "1":
        run_rag_pipeline()
    elif mode_selection == "2":
        run_fine_tuned_model()
    else:
        print("Invalid selection. Please restart and choose either 1 or 2.")

