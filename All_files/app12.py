import streamlit as st
import base64
from final_test_file import run_fine_tuned_model, run_rag_pipeline
import os
import json


def img_to_base64(image_path):
    """Convert image to base64."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None


img_path = "LOGO.webp"
img_base64 = img_to_base64(img_path)
st.set_page_config(
    page_title="AskLaw Legal Assistant",
    page_icon=f"data:image/png;base64,{img_base64}",  # Using the base64-encoded image
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "Get help": None,  # Remove this to avoid displaying the link in the menu
        "Report a bug": None,  # Remove the report bug link as well
        "About": """
            ## AskLaw Legal Assistant


            This assistant helps users analyze legal documents using advanced language models.
        """
    }
)

# Insert custom CSS for glowing effect
st.markdown(
    """
    <style>
    .cover-glow {
        width: 100%;
        height: auto;
        padding: 3px;
        box-shadow: 
            0 0 5px #330000,
            0 0 10px #660000,
            0 0 15px #990000,
            0 0 20px #CC0000,
            0 0 25px #FF0000,
            0 0 30px #FF3333,
            0 0 35px #FF6666;
        position: relative;
        z-index: -1;
        border-radius: 45px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load and display sidebar image


if img_base64:
    st.sidebar.markdown(
        f'<img src="data:LOGO.webp;base64,{img_base64}" class="cover-glow">',
        unsafe_allow_html=True,
    )

st.sidebar.markdown("---")

# Sidebar for Mode Selection
mode = st.sidebar.radio("Select Mode:", options=["RAG Model", "Fine Tuned Model"], index=1)

st.sidebar.markdown("---")

# Display basic interactions
show_basic_info = st.sidebar.checkbox("How to Use the Chatbot", value=True)
if show_basic_info:
    st.sidebar.markdown("""
    ### How to Use the Chatbot
    - **Ask About Virginia Law**: Get answers to specific legal questions related to Virginia's laws and constitution.\n
      Example: *"What rights are protected under Section 1 of the Virginia Constitution?"*
    - **General Legal Questions**: Ask about general legal concepts, definitions, or procedures applicable in Virginia.\n
      Example: *"What is the statute of limitations for civil cases in Virginia?"*
    - **Case Information and Predictions**: Retrieve details about specific cases or list cases based on a legal topic or crime.\n
      Example: *"List manslaughter cases in Virginia."*
    """)
st.sidebar.markdown("---")
img_base64_2 = img_to_base64("github.webp")

st.sidebar.markdown(" \n\n\n\n\n .")


####################### MAIN PAGE ###########################################


# Title
# Main Page Logic
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_query" not in st.session_state:
    st.session_state.user_query = ""
# Insert custom CSS to ensure proper layout and title positioning
st.markdown(
    """
    <style>
    .fixed-title {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;  /* Full width of the page */
        background-color: #121212;  /* Black background for the title */
        color: white;
        padding: 30px;  /* Smaller padding */
        font-size: 22px;  /* Reduced font size for a smaller box */
        text-align: center;
        z-index: 10;
    }
    .content {
        padding-top: 20px;  /* Add space below the fixed title */
        margin-bottom: 20px;  /* Ensure there's space at the bottom for text area */
    }

    .cover-glow {
        width: 100%;
        height: auto;
        padding: 3px;
        box-shadow: 
            0 0 5px #330000,
            0 0 10px #660000,
            0 0 15px #990000,
            0 0 20px #CC0000,
            0 0 25px #FF0000,
            0 0 30px #FF3333,
            0 0 35px #FF6666;
        position: relative;
        z-index: -1;
        border-radius: 45px;
    }
    .stTextArea textarea {
        min-height: 80px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add the fixed title at the top
st.markdown('<div class="fixed-title"><h1>AskLAW Legal Assistant</h1></div>', unsafe_allow_html=True)

st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 15px; padding: 10px;">
        <img src="data:image/webp;base64,{img_base64}" style="width: 50px; height: 50px; border-radius: 50%; box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);">
        <div style="font-size: 16px; font-weight: bold; color: inherit;">Hello! How can I help you today?</div>
    </div>
    """,
    unsafe_allow_html=True,
)
# Text input for user query
chat_history_placeholder = st.empty()  # Placeholder for chat history at the top
latest_response_placeholder = st.empty()  # Placeholder for the latest query response

# Display chat history (if any) but exclude the latest query
if len(st.session_state.chat_history) > 1:  # Show history only for second query onwards
    with chat_history_placeholder.container():
        for idx, entry in enumerate(st.session_state.chat_history[:-1], start=1):  # Exclude the latest entry
            st.markdown(
                f"""
                            <div style="
                                border: 1px solid #444444;
                                border-radius: 8px;
                                padding: 10px;
                                margin-bottom: 10px;
                                background-color: #2C2F33;  /* Dark background */
                                color: #E5E5E5;  /* Light text color */
                                font-family: Arial, sans-serif;
                                font-size: 14px;
                            ">
                                {entry['query']}
                            </div>
                            """,
                unsafe_allow_html=True,
            )
            st.markdown(f" {entry['response']}")
            st.markdown("---")

# Input form for new queries
with st.form(key="query_form", clear_on_submit=True):
    # Set the value of the text_area to be the session state value (will reset on submit)
    user_query = st.text_area("Enter your query here:", value=st.session_state.user_query)

    # If RAG Model is selected, prompt user to upload a JSON file
    uploaded_file = None
    if mode == "RAG Model":
        uploaded_file = st.file_uploader("Upload JSON file", type="json")

    submit_button = st.form_submit_button(label="Submit")
# Handle form submission
if submit_button and (user_query.strip() or uploaded_file):
    response = ""

    # Handle Fine Tuned Model
    if mode == "Fine Tuned Model" and user_query.strip():
        response = run_fine_tuned_model(user_query)

    # Handle RAG Model - process based on file upload or user query
    elif mode == "RAG Model":
        try:
            # If a JSON file is uploaded, process the file content
            if uploaded_file is not None:
                response = run_rag_pipeline(uploaded_file=uploaded_file.name)  # Pass the actual loaded JSON data

            # If no JSON file is uploaded but a query is provided
            elif user_query.strip():
                response = run_rag_pipeline(user_query=user_query)
        except json.JSONDecodeError:
            st.warning("The uploaded file is not a valid JSON. Please upload a correct JSON file.")
            st.stop()

    # Append the latest query and response to chat history
    st.session_state.chat_history.append(
        {"query": user_query if user_query.strip() else "Uploaded File", "response": response})
    st.session_state.user_query = ""
    # Display the latest query and response immediately after the form submission
    with latest_response_placeholder.container():
        if user_query.strip():
            st.markdown(
                f"""
                <div style="
                    border: 1px solid #444444;
                    border-radius: 8px;
                    padding: 10px;
                    margin-bottom: 10px;
                    background-color: #2C2F33;  /* Dark background */
                    color: #E5E5E5;  /* Light text color */
                    font-family: Arial, sans-serif;
                    font-size: 14px;
                ">
                     {user_query}
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:
            file_name = uploaded_file.name
            st.markdown(
                f"""
                <div style="
                    border: 1px solid #444444;
                    border-radius: 8px;
                    padding: 10px;
                    margin-bottom: 10px;
                    background-color: #2C2F33;  /* Dark background */
                    color: #E5E5E5;  /* Light text color */
                    font-family: Arial, sans-serif;
                    font-size: 14px;
                ">
                     {file_name}
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(f"{response}")
        st.markdown("---")

    # Update chat history placeholder for past queries and responses
    if len(st.session_state.chat_history) > 1:  # Show history only for the second query onward
        with chat_history_placeholder.container():
            for idx, entry in enumerate(st.session_state.chat_history[:-1], start=1):  # Exclude the latest entry
                st.markdown(
                    f"""
                               <div style="
                                   border: 1px solid #444444;
                                   border-radius: 8px;
                                   padding: 10px;
                                   margin-bottom: 10px;
                                   background-color: #2C2F33;  /* Dark background */
                                   color: #E5E5E5;  /* Light text color */
                                   font-family: Arial, sans-serif;
                                   font-size: 14px;
                               ">
                                    {entry['query']}
                               </div>
                               """,
                    unsafe_allow_html=True,
                )
                st.markdown(f" {entry['response']}")
                st.markdown("---")
    user_query = ""
elif submit_button:
    st.warning("Please enter a query or upload a file before submitting.")
st.markdown(
    f"""
    <div style="position: fixed; bottom: 20px; right: 20px; text-align: center;">
        <a href="https://github.com/Saikrishna-Paila/Legal_Document_Analysis_using_LLM" target="_blank">
            <img src="data:image/webp;base64,{img_base64_2}" width="40" height="40" />
        </a>
    </div>
    """, unsafe_allow_html=True
)