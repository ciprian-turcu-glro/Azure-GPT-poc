import os
import time
import streamlit as st
import openai

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ["OPENAI_API_KEY"]


load_dotenv()

openai.api_type = "azure"
openai.api_base = "https://oai-playground-canadaeast-001.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")


def initialise_variables(session_variables):
    for variable in session_variables:
        if variable["name"] not in st.session_state:
            # print(variable["name"], ":", variable["value"])
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


def generate_custom_text_for_simple_query_expansion(option):
    if is_option_match(option, 1):
        return "when was the  an assessment of the useful lives of our server and network equipment completed ? which month was that in?"
    elif is_option_match(option, 2):
        return "What was the total annual revenue for 2022?"
    elif is_option_match(option, 3):
        return "how many people do we want to recruit?"
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
                                    print(content)
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
            "content": "You are a helpful expert financial research assistant. Provide an example answer to the given question, that might be found in a document like an annual report. ",
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
    print("------------------------------------------")
    print("------------------------------------------")
    print(content)
    print("------------------------------------------")
    print("------------------------------------------")
    return content


# RAG
def apply_rag(
    query,
    pdf="data/2022_Annual_Report.pdf",
    n_results=5,
    query_include=["metadatas", "documents", "distances"],
):
    if st.session_state.submitted:
        # Step 1: load the text content from the pdf
        #
        from pypdf import PdfReader

        reader = PdfReader(pdf)
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
        ids = [str(i) for i in range(len(token_split_texts))]

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
