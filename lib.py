import os
import time
import streamlit as st
import openai
from dotenv import load_dotenv
from pypdf import PdfReader

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


def prompt_request(pre_prompt_text, custom_messages=[], prompt=""):
    if len(custom_messages) < 1:
        custom_messages.append(
            {
                "role": "system",
                "content": "You are an AI assistant that helps people find information.",
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


def word_wrap(string, n_chars=72):
    # Wrap a string at the next space after n_chars
    if len(string) < n_chars:
        return string
    else:
        return string[:n_chars].rsplit(' ', 1)[0] + '\n' + word_wrap(string[len(string[:n_chars].rsplit(' ', 1)[0])+1:], n_chars)
    

def read_pdf(filename: str):
    st.session_state.submitted = False if st.session_state.submitted == True else True
    reader = PdfReader(filename)
    pdf_texts = [p.extract_text().strip() for p in reader.pages]

    # Filter the empty strings
    pdf_texts = [text for text in pdf_texts if text]
    st.session_state.pdf_text = pdf_texts
    