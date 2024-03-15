import openai
import streamlit as st
from lib import *
from extract_lib import *

# initialized variables state names and values
session_variables = get_session_variables()
run_prompt = False
rag_story = ""

# method to set the initialised variables to state variables
initialise_variables(session_variables)
# ------------------------------------------------------------------------------------------
# --- RAG START
# ------------------------------------------------------------------------------------------
if st.session_state.submitted:
    # Step 1: load the text content from the pdf
    #
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    pdf_texts = [p.extract_text().strip() for p in reader.pages]

    # Filter the empty strings
    # --
    pdf_texts = [text for text in pdf_texts if text]

    #
    # Step 2: Split by character and token
    #

    # import text and token splitter functions
    # --
    from langchain.text_splitter import (
        RecursiveCharacterTextSplitter,
        SentenceTransformersTokenTextSplitter,
    )

    # Split in text chunks:
    # --
    character_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
    )
    character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

    # Turn text chunks into newly split token chunks
    # --
    token_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=0, tokens_per_chunk=256
    )

    token_split_texts = []
    for text in character_split_texts:
        token_split_texts += token_splitter.split_text(text)

    #
    # Step 3: Apply POOLING to token chunks to create optimised Sentance Embeddings
    #
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    embedding_function = SentenceTransformerEmbeddingFunction()

    # Create collection and id's for chunks in collection
    # --
    chroma_client = chromadb.Client()
    try:
        chroma_collection = chroma_client.get_collection("microsoft_annual_report_2022")
        chroma_client.delete_collection("microsoft_annual_report_2022")
        chroma_collection = chroma_client.create_collection(
            "microsoft_annual_report_2022", embedding_function=embedding_function
        )
    except:
        chroma_collection = chroma_client.create_collection(
            "microsoft_annual_report_2022", embedding_function=embedding_function
        )

    # list of stringed id's for identifying the chunks of documents in the collection
    # --
    ids = [str(i) for i in range(len(token_split_texts))]

    chroma_collection.add(ids=ids, documents=token_split_texts)
    chroma_collection.count()

    #
    # Step 4: query the collection with the prompt data:
    query = "What was the total revenue?"

    results = chroma_collection.query(query_texts=[query], n_results=5)
    retrieved_documents = results["documents"][0]

    # for document in retrieved_documents:
    # document
#

# ------------------------------------------------------------------------------------------
#  --- RAG END
# ------------------------------------------------------------------------------------------


st.title("Basic RAG")


rag_option = st.selectbox(
    "Automatic sugestions:",
    (st.session_state.rag_options),
    key=1,
)
prompt_value = generate_custom_text_for_simple_rag(rag_option)
prompt_value = st.text_input(
    key="field_prompt_value",
    value=prompt_value,
    label="your prompt:",
)


if st.session_state.submitted:
    #
    messages = [
        {
            "role": "system",
            "content": "You are a helpful expert financial research assistant. Your users are asking questions about information contained in an annual report."
            "You will be shown the user's question, and the relevant information from the annual report. Answer the user's question using only this information.",
        },
        {
            "role": "user",
            "content": f"Question: {prompt_value}. \n Information: {retrieved_documents}",
        },
    ]
    #
    custom_messages = custom_messages_generating(
        rag_story, messages, prompt=prompt_value
    )
    completion = openai.ChatCompletion.create(
        engine="Chip-GPT4-32k",
        messages=custom_messages,
        temperature=0.7,
        stream=True,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
    )
st.button(
    label="Send" if st.session_state.submitted == False else "Clear",
    key=4,
    on_click=lambda: on_run_clicked(),
)
res_container = st.empty()
if st.session_state.submitted:
    process_response(completion)
