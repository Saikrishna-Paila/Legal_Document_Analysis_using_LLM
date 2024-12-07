

# AI Meets Law: Transforming Legal Research with RAG, Fine-Tuning, and Few-Shot Learning

## Project Overview
This project brings together state-of-the-art AI techniques like **Retrieval-Augmented Generation (RAG)**, **Fine-Tuning**, and **Few-Shot Learning (FSL)** to revolutionize the way legal research is conducted. By leveraging historical legal case data and constitutional texts, it enables lawyers and legal professionals to retrieve, analyze, and interpret legal information efficiently.

The system is designed to address the challenges of legal research, such as sifting through vast amounts of data and interpreting nuanced language. By combining retrieval capabilities with a fine-tuned language model, it provides insights, summaries, and even predictions with high accuracy and domain-specific expertise.

---

## Team Members
- **Saikrishna Paila** 
- **Aneri Patel** 
- **Srivallabh Siddharth N** 

---

## Features
- **RAG Pipeline**: Combines embeddings and FAISS for fast and accurate document retrieval, supplemented by text summarization.
- **Fine-Tuning**: Customizes a GPT-4 model for legal tasks, improving its understanding of legal jargon and case-specific scenarios.
- **Few-Shot Learning (FSL)**: Leverages small curated datasets to provide detailed and nuanced responses.
- **Streamlit App**: Offers an easy-to-use graphical interface for both individual queries and bulk processing.
- **Batch Processing**: Handles JSON uploads to process multiple cases simultaneously, extracting key insights efficiently.

---

## Concepts Used

### **Retrieval-Augmented Generation (RAG)**
RAG combines document retrieval with generative language models. It first retrieves the most relevant documents for a user query using FAISS and embeddings, then passes these documents as context to a language model for generating a detailed response.

### **Few-Shot Learning (FSL)**
Few-shot learning allows the model to adapt to specific legal contexts by learning from a small set of curated examples. These examples guide the model to provide accurate, domain-specific responses.

### **Fine-Tuning**
Fine-tuning enhances a pre-trained GPT-4 model to specialize in legal tasks. This process adapts the model’s responses to legal terminologies, case formats, and reasoning patterns.

### **FAISS (Facebook AI Similarity Search)**
FAISS enables efficient similarity searches among vectorized documents. It ensures that the system can quickly find the most relevant legal documents by embedding both user queries and case data into a shared vector space.

### **Sentence Transformers**
Sentence Transformers are used to convert textual data (e.g., case opinions) into embeddings, which are numerical representations capturing the semantic meaning of text. These embeddings are crucial for similarity-based retrieval.

### **Streamlit**
Streamlit provides a user-friendly interface for interacting with the system. Users can input queries, upload case files, and visualize results with minimal technical expertise.

---

## Tools and Technologies
- **Python**: Core programming language used for the pipelines and app.
- **LangChain**: Framework for chaining and managing LLM-based workflows.
- **FAISS**: For high-speed similarity searches in vectorized data.
- **Sentence Transformers**: Generates embeddings for text similarity tasks.
- **Streamlit**: Builds the web-based interactive application.
- **Pandas and NumPy**: Used for data preprocessing and management.
- **Matplotlib**: For visualizing patterns and trends in legal data.

---

## Requirements

To run this project, the following setup is required:

- **Python Version**: 3.8 or higher
- **Dependencies**:
  - `langchain`
  - `sentence-transformers`
  - `faiss`
  - `streamlit`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `torch` (for working with transformers and fine-tuning)

Install all required dependencies using:
```bash
pip install -r requirements.txt
```

Hardware requirements:
- At least **16GB RAM** for handling large embeddings and datasets efficiently.

---

## File Structure
- **`0840-01_arguments_only.json`**: Sample dataset containing legal arguments and metadata for testing.
- **`learning_examples.json`**: Few-shot learning examples for guiding the fine-tuned model.
- **`RAG.py`**: Implements the retrieval-augmented generation pipeline for query handling and summarization.
- **`RAG+FSL+Fine_tune.py`**: Fine-tunes the GPT-4 model and integrates FSL with the RAG pipeline.
- **`app.py`**: Streamlit-based user interface for querying and bulk case analysis.

---

## Important Configuration

### Add OpenAI API Key
1. Create a `.env` file in the root directory of the project.
2. Add your OpenAI API key in the following format:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```
3. Ensure that the `.env` file is loaded correctly by verifying the API key in the scripts.

### Fine-Tune Your Own Model
- To fine-tune your GPT model, follow the instructions in the video tutorial:
  [Fine-Tune Your LLMs](https://www.linkedin.com/learning/fine-tune-your-llms/introduction-to-fine-tuning-llms?u=74651410)
- This process involves preparing your dataset (e.g., `output.jsonl`), uploading it to OpenAI, and fine-tuning the model for legal-specific tasks.

---

## How to Run

### 1. RAG Pipeline
To run the retrieval-augmented generation (RAG) pipeline:
```bash
python RAG.py
```
- **Functionality**: Embeds user queries, retrieves top documents using FAISS, and generates summaries.

---

### 2. Fine-Tuning and Few-Shot Learning
To fine-tune the model and integrate few-shot learning:
```bash
python RAG+FSL+Fine_tune.py
```
- **Functionality**: Customizes GPT-4 for handling legal queries with greater accuracy and relevance.
- **Output**: Fine-tuned predictions and enhanced contextual responses.

---

### 3. Streamlit App
To launch the graphical interface:
```bash
streamlit run app.py
```
- **Modes**:
  - **Query Mode**: Allows users to input queries and receive insights instantly.
  - **Test Case Mode**: Supports JSON file uploads for batch processing and analysis.

---

## Workflow

### **Query Mode**
1. User enters a query (e.g., "What are the free speech rights in Virginia?").
2. The query is embedded and searched in the FAISS index for relevant legal documents.
3. Retrieved documents are summarized if needed and passed to the model.
4. The system generates a concise, contextually relevant response.

---

### **Test Case Mode**
1. User uploads a JSON file containing multiple legal cases.
2. The system:
   - Extracts and combines case opinions into a single query (if applicable).
   - Analyzes each case using the RAG pipeline.
   - Summarizes results and presents them to the user.

---

### **Fine-Tuning Workflow**
1. **Prepare Data**:
   - Preprocess data files, including `learning_examples.json`, to ensure clean and structured input.
   - Generate vector embeddings using `SentenceTransformer`.

2. **Train**:
   - Fine-tune GPT-4 using curated few-shot examples for legal tasks.
   - Optimize for domain-specific language and reasoning.

3. **Validate**:
   - Test the fine-tuned model with unseen queries.
   - Adjust the training process to improve performance where necessary.

4. **Integrate**:
   - Use the FAISS index to retrieve relevant documents.
   - Combine context with few-shot examples to guide the fine-tuned model.

---

## Future Enhancements
- **Dataset Expansion**: Add more cases and legal documents to improve system coverage.
- **Model Optimization**: Streamline fine-tuning for faster training and better results.
- **API Development**: Create APIs for seamless third-party integration.
- **Interactive Visualizations**: Build dashboards for exploring case patterns and trends.

---


