import os
import time
import streamlit as st
import openai
from extract_lib import save_to_file

from helper_utils import load_chroma, word_wrap, project_embeddings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import numpy as np
import umap
from tqdm import tqdm
import matplotlib.pyplot as plt
import fitz

import torch

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ["OPENAI_API_KEY"]


load_dotenv()

openai.api_type = "azure"
openai.api_base = "https://oai-playground-canadaeast-001.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")


def getPyMuPDF(file):
    doc = fitz.open(file)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def initialise_variables(session_variables):
    for variable in session_variables:
        if variable["name"] not in st.session_state:
            st.session_state[variable["name"]] = variable["value"]


def on_run_clicked():
    st.session_state.submitted = False if st.session_state.submitted == True else True
    st.session_state.final_response = ""
    if st.session_state.submitted == False:
        st.session_state.field_textarea_value = ""


def is_option_match(option, index):
    return st.session_state.rag_options.index(option) == index


def generate_custom_text(option):
    if is_option_match(option, 1):
        return "Why didn't Anna go out for a walk on tuesday?"
    elif is_option_match(option, 2):
        return "How many fruit did Anna start her walk with?"
    elif is_option_match(option, 3):
        return "Where was Anna going?"
    else:
        return ""


def generate_custom_text_for_simple_rag(option):
    if is_option_match(option, 1):
        return "What was the total annual revenue in 2022?"

    else:
        return ""


def generate_custom_text(option):
    if is_option_match(option, 1):
        return "when was the  an assessment of the useful lives of our server and network equipment completed ? which month was that in?"
    elif is_option_match(option, 2):
        return "What was the total annual revenue for 2022?"
    elif is_option_match(option, 3):
        return "how many people do we want to recruit?"
    elif is_option_match(option, 4):
        return "what is the purpose of the Microsoft Intelligent Data 5 Platform in the context of the annual report ?"
    else:
        return ""


def initial_object_prompts():
    custom_messages = []

    custom_messages.append(
        {
            "role": "system",
            "content": "You are an AI assistant that helps people find information.",
        }
    )
    custom_messages.append(
        {
            "role": "user",
            "content": "",
        },
    )
    return custom_messages


def custom_messages_generating(
    pre_prompt_text=None,
    custom_messages=[],
    prompt="",
    system_content="You are an AI assistant that helps people find information.",
):
    if len(custom_messages) == 0:
        custom_messages.append(
            {
                "role": "system",
                "content": system_content,
            }
        )
        custom_messages.append(
            {
                "role": "user",
                "content": pre_prompt_text,
            },
        )
        custom_messages.append(
            {
                "role": "user",
                "content": prompt,
            },
        )

    else:
        custom_messages.append(
            {
                "role": "user",
                "content": pre_prompt_text,
            },
        )

    return custom_messages


def process_response(response):
    if st.session_state.submitted == True:
        try:
            with st.empty():
                final_response = ""
                for chunk in response:
                    choices = chunk["choices"]
                    isChoicesGreaterThanZero = len(choices) > 0
                    if isChoicesGreaterThanZero:
                        delta = choices[0].delta
                        isDeltaExists = len(delta) > 0
                        if isDeltaExists and "content" in delta:
                            content = delta["content"]
                            isContentExists = len(content) > 0
                            if isContentExists:
                                if (
                                    isChoicesGreaterThanZero
                                    and isDeltaExists
                                    and isContentExists
                                ):
                                    # final_response = final_response + content
                                    old_word = ""
                                    for word in content.split(" "):
                                        final_response += " " + word
                                        if word != old_word:
                                            st.write(final_response)
                                            # st.session_state.final_response = (
                                            #     st.session_state.final_response + word
                                            #     )
                                            time.sleep(0.1)
                                            old_word = word
        except Exception as e:
            print("OpenAI Response Streaming error: " + str(e))


def process_terminal_response(response):
    try:
        final_response = ""
        print("\nAnswer: \n")
        for chunk in response:
            choices = chunk["choices"]
            isChoicesGreaterThanZero = len(choices) > 0
            if isChoicesGreaterThanZero:
                delta = choices[0].delta
                isDeltaExists = len(delta) > 0
                if isDeltaExists and "content" in delta:
                    content = delta["content"]
                    isContentExists = len(content) > 0
                    if isContentExists:
                        if (
                            isChoicesGreaterThanZero
                            and isDeltaExists
                            and isContentExists
                        ):
                            final_response = final_response + content
                            # st.write(final_response)
                            # st.session_state.final_response = (
                            #     st.session_state.final_response + content
                            # )
        print(final_response)
    except Exception as e:
        print("OpenAI Response Streaming error: " + str(e))
        # return 503


def openai_prompt_request(
    query, model="Chip-GPT4-32k", custom_messages=[], stream=False
):
    default_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. You help users answer questions from the provided manual.",
        },
        {"role": "user", "content": query},
    ]

    response = openai.ChatCompletion.create(
        engine=model,
        messages=default_messages if len(custom_messages) == 0 else custom_messages,
        temperature=0.7,
        stream=stream,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
    )
    content = response.choices[0].message.content

    return content


# RAG
def apply_rag(
    query,
    pdf="./data/BD-D100_D120GV_XGV.pdf",
    n_results=5,
    query_include=["metadatas", "documents", "distances"],
):
    if st.session_state.submitted:
        # Step 1: load the text content from the pdf
        #
        # from pypdf import PdfReader

        # reader = PdfReader(pdf)
        pdf_texts = read_pdf_with_pyMuPDF(pdf)
        # pdf_texts = [p.extract_text().strip() for p in reader.pages]

        # Filter the empty strings
        # --
        # pdf_texts = [text for text in pdf_texts if text]
        # print(pdf_texts)

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
        from chromadb.utils.embedding_functions import (
            SentenceTransformerEmbeddingFunction,
        )

        embedding_function = SentenceTransformerEmbeddingFunction()

        # Create collection and id's for chunks in collection
        # --
        chroma_client = chromadb.Client()
        if chroma_client.count_collections() > 0:
            chroma_client.delete_collection("custom_collection_1")
        chroma_collection = chroma_client.create_collection(
            "custom_collection_1", embedding_function=embedding_function
        )

        # list of stringed id's for identifying the chunks of documents in the collection
        # --
        print("--------------token_split_texts", token_split_texts)
        ids = [str(i) for i in range(len(token_split_texts))]
        save_to_file(
            " ".join(map(str, token_split_texts)), "./data/chunks/chunks_test.txt"
        )

        chroma_collection.add(ids=ids, documents=token_split_texts)
        chroma_collection.count()

        #
        # Step 4: query the collection with the prompt data:
        # query = "What was the total revenue?"

        results = chroma_collection.query(
            query_texts=[query], n_results=n_results, include=query_include
        )
        retrieved_documents = results["documents"][0]
        print()
        return retrieved_documents


def embedding_adapters(
    file="data/D100_D120GV_XGV.pdf", collection_name="microsoft_annual_report_2022"
):
    embedding_function = SentenceTransformerEmbeddingFunction()

    chroma_collection = load_chroma(
        filename=file,
        collection_name=collection_name,
        embedding_function=embedding_function,
    )
    chroma_collection.count()
    # embeddings
    embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]
    umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
    projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

    all_generated_queries = generate_queries()
    for query in all_generated_queries:
        print(query)
    query_results = chroma_collection.query(
        query_texts=all_generated_queries,
        n_results=10,
        include=["documents", "embeddings"],
    )
    retrieved_documents = query_results["documents"]
    retrieved_embeddings = query_results["embeddings"]
    query_embeddings = embedding_function(all_generated_queries)

    adapter_query_embeddings = []
    adapter_doc_embeddings = []
    adapter_labels = []

    for q, query in enumerate(tqdm(all_generated_queries)):
        for d, document in enumerate(retrieved_documents[q]):
            adapter_query_embeddings.append(query_embeddings[q])
            adapter_doc_embeddings.append(retrieved_embeddings[q][d])
            adapter_labels.append(evaluate_results(query, document))
    print(len(adapter_labels))
    adapter_query_embeddings = torch.Tensor(np.array(adapter_query_embeddings))
    adapter_doc_embeddings = torch.Tensor(np.array(adapter_doc_embeddings))
    adapter_labels = torch.Tensor(np.expand_dims(np.array(adapter_labels), 1))

    dataset = torch.utils.data.TensorDataset(
        adapter_query_embeddings, adapter_doc_embeddings, adapter_labels
    )

    # Initialize the adaptor matrix
    mat_size = len(adapter_query_embeddings[0])
    adapter_matrix = torch.randn(mat_size, mat_size, requires_grad=True)

    min_loss = float("inf")
    best_matrix = None

    for epoch in tqdm(range(100)):
        for query_embedding, document_embedding, label in dataset:
            loss = mse_loss(query_embedding, document_embedding, adapter_matrix, label)

            if loss < min_loss:
                min_loss = loss
                best_matrix = adapter_matrix.clone().detach().numpy()

            loss.backward()
            with torch.no_grad():
                adapter_matrix -= 0.01 * adapter_matrix.grad
                adapter_matrix.grad.zero_()

    print(f"Best loss: {min_loss.detach().numpy()}")

    test_vector = torch.ones((mat_size, 1))
    scaled_vector = np.matmul(best_matrix, test_vector).numpy()

    plt.bar(range(len(scaled_vector)), scaled_vector.flatten())
    plt.show()

    query_embeddings = embedding_function(all_generated_queries)
    adapted_query_embeddings = np.matmul(best_matrix, np.array(query_embeddings).T).T

    projected_query_embeddings = project_embeddings(query_embeddings, umap_transform)
    projected_adapted_query_embeddings = project_embeddings(
        adapted_query_embeddings, umap_transform
    )
    # Plot the projected query and retrieved documents in the embedding space
    plt.figure()
    plt.scatter(
        projected_dataset_embeddings[:, 0],
        projected_dataset_embeddings[:, 1],
        s=10,
        color="gray",
    )
    plt.scatter(
        projected_query_embeddings[:, 0],
        projected_query_embeddings[:, 1],
        s=150,
        marker="X",
        color="r",
        label="original",
    )
    plt.scatter(
        projected_adapted_query_embeddings[:, 0],
        projected_adapted_query_embeddings[:, 1],
        s=150,
        marker="X",
        color="green",
        label="adapted",
    )

    plt.gca().set_aspect("equal", "datalim")
    plt.title("Adapted Queries")
    plt.axis("off")
    plt.legend()


def generate_queries(model="Chip-GPT4-32k"):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. You help users answer questions from the provided context. "
            "Suggest 10 to 15 short questions that are important to ask when analyzing the givem context. "
            "Do not output any compound questions (questions with multiple sentences or conjunctions)."
            "Output each question on a separate line divided by a newline.",
        },
    ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    content = content.split("\n")
    return content


def evaluate_results(query, statement, model="Chip-GPT4-32k"):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. You help users answer questions from the provided context."
            "For the given query statement, evaluate whether the following satement is relevant."
            "Output only 'yes' or 'no'.",
        },
        {"role": "user", "content": f"Query: {query}, Statement: {statement}"},
    ]

    response = openai.ChatCompletion.create(
        model=model, messages=messages, max_tokens=1
    )
    content = response.choices[0].message.content
    if content == "yes":
        return 1
    return -1


def model(query_embedding, document_embedding, adaptor_matrix):
    updated_query_embedding = torch.matmul(adaptor_matrix, query_embedding)
    return torch.cosine_similarity(updated_query_embedding, document_embedding, dim=0)


def mse_loss(query_embedding, document_embedding, adaptor_matrix, label):
    return torch.nn.MSELoss()(
        model(query_embedding, document_embedding, adaptor_matrix), label
    )


def get_session_variables():
    return [
        {"name": "submitted", "value": False},
        {"name": "field_textarea_value", "value": ""},
        {"name": "final_response", "value": ""},
        {"name": "rag_selectbox", "value": ""},
        {
            "name": "rag_options",
            "value": ["Custom", "Option 1", "Option 2", "Option 3", "Option 4"],
        },
        {
            "name": "reader_options",
            "value": ["pdfminer", "pyMuPDF"],
        },
    ]


# extracting the text from a file path
def read_pdf_with_pyMuPDF(file_path):
    text = ""
    try:
        # Open the PDF file
        with fitz.open(file_path) as pdf_file:
            # Iterate through each page
            for page_number in range(pdf_file.page_count):
                # Get the current page
                page = pdf_file.load_page(page_number)
                # Extract text from the current page
                text += page.get_text()
    except Exception as e:
        print(f"Error reading PDF file: {e}")

    return text
