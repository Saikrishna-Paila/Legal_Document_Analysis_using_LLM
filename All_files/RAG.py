import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from sentence_transformers import SentenceTransformer
import faiss
from langchain.schema import HumanMessage
from langchain.memory import ConversationBufferMemory

# Update imports for LangChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = "add your api key"
import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from sentence_transformers import SentenceTransformer
import faiss
from langchain.schema import HumanMessage
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Update imports for LangChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = "add your api OPENAI_API_KEY"
# Function to load JSON files
def load_json_files(directory: str) -> List[Dict]:
    """
    Load case data from JSON files in a directory and enrich metadata.
    """
    files = Path(directory).glob("*.json")
    cases = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            case = json.load(f)
            for opinion in case.get("casebody", {}).get("opinions", []):
                decision_date = case.get("decision_date", "")
                year = decision_date[:4] if decision_date else "Unknown"
                enriched_case = {
                    "id": case.get("id", ""),
                    "case_id": case.get("id", ""),
                    "case_name": case.get("name", ""),
                    "decision_date": decision_date,
                    "jurisdiction": case.get("jurisdiction", {}).get("name", "Unknown"),
                    "text": opinion.get("text", "").strip(),
                    "year": year,
                    "tags": case.get("tags", []),
                }
                cases.append(enriched_case)
    return cases

# Fine-tuned model loader
def load_fine_tuned_model(model_name="ft:gpt-4o-mini-2024-07-18:personal::AabQd3B8"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# Generate response using the fine-tuned model
def query_fine_tuned_model(query: str, context: str, tokenizer, model, max_length=512):
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=max_length, truncation=True)
    outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# Create RAG pipeline with fine-tuned model
def create_rag_with_fine_tuned_pipeline(index, metadata, fine_tuned_model_name="ft:gpt-4o-mini-2024-07-18:personal::AabQd3B8"):
    tokenizer, model = load_fine_tuned_model(fine_tuned_model_name)

    # Create RAG pipeline
    def filter_retrieved_docs(query: str, top_k=5) -> List[Dict]:
        query_embedding = SentenceTransformer("all-MiniLM-L6-v2").encode([query])[0]
        distances, indices = index.search(np.array([query_embedding]), top_k)

        relevant_docs = []
        for i, idx in enumerate(indices[0]):
            if distances[0][i] > 0.1:  # Adjust threshold as needed
                doc = metadata[idx]
                relevant_docs.append(doc)

        return relevant_docs

    def query_combined(query: str, use_rag=True, use_fine_tuned=True) -> str:
        # Retrieve documents using RAG
        context = ""
        if use_rag:
            relevant_docs = filter_retrieved_docs(query, top_k=10)
            context = "\n\n".join([
                f"{doc.get('case_name', doc.get('id', 'Document'))}:\n{doc.get('text', 'No text available')[:1500]}"
                for doc in relevant_docs
            ])
            if not relevant_docs:
                return "No relevant information found in the knowledge base."

        # Generate response using the fine-tuned model
        fine_tuned_response = ""
        if use_fine_tuned:
            fine_tuned_response = query_fine_tuned_model(query, context, tokenizer, model)

        # Combine responses if both are used
        if use_rag and use_fine_tuned:
            combined_response = f"RAG Context:\n{context}\n\nFine-Tuned Model Response:\n{fine_tuned_response}"
            return combined_response
        elif use_rag:
            return context
        elif use_fine_tuned:
            return fine_tuned_response

    return query_combined

if __name__ == "__main__":
    directory = "data"  # Replace with your actual directory for JSON case files
    va_constitution_path = "va_constitution.json"
    pkl_file = "finaldata1.pkl"

    # Load case data
    cases = load_json_files(directory)

    # Load VA Constitution file
    def load_va_constitution(file_path: str) -> List[Dict]:
        """
        Load the VA Constitution file and process it into sections.
        """
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

    va_sections = load_va_constitution(va_constitution_path)

    # Combine cases and VA Constitution sections
    all_data = cases + va_sections

    # Create embeddings and save to pickle file
    def create_embeddings(data: List[Dict], model_name="all-MiniLM-L6-v2", pkl_file="finaldata1.pkl") -> None:
        """
        Generate embeddings for data (cases or other texts), skipping duplicates and invalid entries, and save to pickle.
        """
        model = SentenceTransformer(model_name)
        if os.path.exists(pkl_file):
            with open(pkl_file, "rb") as f:
                combined_data = pickle.load(f)
        else:
            combined_data = []

        existing_ids = {entry["metadata"].get("id") for entry in combined_data if "metadata" in entry}
        new_data = []
        for entry in data:
            entry_id = entry.get("id", "")
            if entry_id in existing_ids or not entry.get("text", "").strip():
                continue
            try:
                embedding = model.encode(entry["text"].strip(), normalize_embeddings=True)
                new_data.append({"embedding": embedding, "metadata": entry})
            except Exception as e:
                print(f"Error generating embedding for entry {entry.get('id', 'Unknown')}: {e}")

        combined_data.extend(new_data)
        with open(pkl_file, "wb") as f:
            pickle.dump(combined_data, f)

    create_embeddings(all_data, pkl_file=pkl_file)

    # Load FAISS index and metadata
    def load_faiss_index(pkl_file="finaldata1.pkl") -> Tuple[faiss.IndexFlatIP, List[Dict]]:
        """
        Load embeddings and metadata from the pickle file and create a FAISS index.
        """
        with open(pkl_file, "rb") as f:
            combined_data = pickle.load(f)

        embeddings = np.vstack([entry["embedding"] for entry in combined_data if "embedding" in entry])
        metadata = [entry["metadata"] for entry in combined_data if "metadata" in entry]

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return index, metadata

    index, metadata = load_faiss_index(pkl_file)

    # Initialize the pipeline with the fine-tuned model
    query_combined = create_rag_with_fine_tuned_pipeline(
        index, metadata, fine_tuned_model_name="ft:gpt-4o-mini-2024-07-18:personal::AabQd3B8"
    )

    while True:
        user_input = input("Enter your query (type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        response = query_combined(user_input, use_rag=True, use_fine_tuned=True)
        print(f"Answer:\n{response}")


def load_va_constitution(file_path: str) -> List[Dict]:
    """
    Load the VA Constitution file and process it into sections.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # Split the text into sections based on headings or articles.
    # Adjust the splitting logic based on the actual structure of your VA Constitution file.
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
    """
    Generate embeddings for data (cases or other texts), skipping duplicates and invalid entries, and save to pickle.
    """
    model = SentenceTransformer(model_name)
    if os.path.exists(pkl_file):
        with open(pkl_file, "rb") as f:
            combined_data = pickle.load(f)
    else:
        combined_data = []

    existing_ids = {entry["metadata"].get("id") for entry in combined_data if "metadata" in entry}
    new_data = []
    for entry in data:
        entry_id = entry.get("id", "")
        if entry_id in existing_ids or not entry.get("text", "").strip():
            continue
        try:
            embedding = model.encode(entry["text"].strip(), normalize_embeddings=True)
            new_data.append({"embedding": embedding, "metadata": entry})
        except Exception as e:
            print(f"Error generating embedding for entry {entry.get('id', 'Unknown')}: {e}")

    combined_data.extend(new_data)
    with open(pkl_file, "wb") as f:
        pickle.dump(combined_data, f)

def extract_named_entities(text: str) -> List[str]:
    """
    Extract named entities from a query using NLTK.
    """
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    chunks = ne_chunk(pos_tags, binary=False)

    named_entities = []
    for chunk in chunks:
        if isinstance(chunk, Tree):
            named_entity = " ".join(c[0] for c in chunk)
            named_entities.append(named_entity)
    return named_entities

def load_faiss_index(pkl_file="finaldata1.pkl") -> Tuple[faiss.IndexFlatIP, List[Dict]]:
    """
    Load embeddings and metadata from the pickle file and create a FAISS index.
    """
    with open(pkl_file, "rb") as f:
        combined_data = pickle.load(f)

    embeddings = np.vstack([entry["embedding"] for entry in combined_data if "embedding" in entry])
    metadata = [entry["metadata"] for entry in combined_data if "metadata" in entry]

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, metadata

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
You are a legal assistant tasked with answering questions based on the following context from the Virginia Constitution and relevant case law. If the context does not provide the answer, use your own knowledge to answer the question.

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
You are a legal assistant tasked with answering questions based on the following context from the Virginia Constitution and relevant case law. If the context does not provide the answer, use your own knowledge to answer the question.

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

def summarize_text(text, max_tokens=500):
    """
    Summarize a given text to fit within token constraints while retaining key arguments.
    """
    llm = ChatOpenAI(model_name="gpt-4", max_tokens=max_tokens, temperature=0)
    prompt = f"Summarize the following legal arguments concisely:\n\n{text}"
    return llm.predict(prompt).strip()


if __name__ == "__main__":
    directory = "Case 1 JSON"
    pkl_file = "finaldata1.pkl"

    # Load case data
    cases = load_json_files(directory)

    # Load VA Constitution file (path is fixed)
    va_constitution_path = "va_constitution.json"
    va_sections = load_va_constitution(va_constitution_path)

    # Combine cases and VA Constitution sections
    all_data = cases + va_sections

    # Create embeddings and save to pickle file
    create_embeddings(all_data, pkl_file=pkl_file)

    # Load FAISS index and metadata
    index, metadata = load_faiss_index(pkl_file)
    query_rag, memory, context_store = create_rag_pipeline(index, metadata)

    mode = "query"  # Start in query mode

    while True:
        if mode == "query":
            user_input = input("Enter your query (type 'esc' to switch to test case mode, 'exit' to quit): ")
            if user_input.lower() == "exit":
                break
            elif user_input.lower() == "esc":
                mode = "test_case"
                continue
            else:
                if user_input.strip() == "":
                    print("Please enter a valid query.")
                    continue
                # If user wants to refer to last query and answer
                if user_input.lower() in ["what was my last query?", "what was the last answer?", "refer to last"]:
                    if context_store["last_query"] and context_store["last_answer"]:
                        print(f"Your last query was: {context_store['last_query']}")
                        print(f"The answer was: {context_store['last_answer']}")
                    else:
                        print("No previous query found.")
                    continue

                response = query_rag(user_input)
                print(f"Answer: {response}")

        elif mode == "test_case":
            test_file_path = input("Enter the path to the test file (type 'esc' to switch back, 'exit' to quit): ")
            if test_file_path.lower() == "exit":
                break
            elif test_file_path.lower() == "esc":
                mode = "query"
                continue
            else:
                try:
                    with open(test_file_path, 'r', encoding='utf-8') as f:
                        test_file_data = json.load(f)

                    # Combine the text of all opinions into one query
                    combined_query = " ".join([
                        opinion.get("text", "").strip()
                        for opinion in test_file_data["casebody"]["opinions"]
                        if opinion.get("text")
                    ])

                    # Ensure the query is summarized if too large
                    if len(combined_query) > 2000:  # Adjust this threshold as needed
                        print("Summarizing the test case to fit within processing limits...")
                        combined_query = summarize_text(combined_query)

                    # Get the expected verdict for the entire test file
                    response = query_rag(combined_query)
                    print(f"Expected Verdict for Test File: {response}")

                    # After processing, switch back to query mode
                    mode = "query"

                except Exception as e:
                    print(f"Error processing test file: {str(e)}")
                    # Optionally, stay in test_case mode to allow retry
                    continue

    print("Exiting the program. Goodbye!")
