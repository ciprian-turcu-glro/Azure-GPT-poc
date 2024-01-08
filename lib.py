import os
import streamlit as st
import openai
from dotenv import load_dotenv

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
    print(st.session_state.field_textarea_value)
    st.session_state.final_response = ""
    if st.session_state.submitted == False:
        st.session_state.field_textarea_value = ""


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
            "content": st.session_state.field_textarea_value,
        },
    )
    return custom_messages


def prompt_request(prompt_text):
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
            "content": prompt_text,
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
                                    final_response = final_response + content
                                    st.write(final_response)
                                    # st.session_state.final_response = (
                                    #     st.session_state.final_response + content
                                    # )
        except Exception as e:
            print("OpenAI Response Streaming error: " + str(e))


def process_terminal_response(response):
    try:
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
                            final_response = final_response + content
                            # st.write(final_response)
                            # st.session_state.final_response = (
                            #     st.session_state.final_response + content
                            # )
        print("\nAnswer: \n" + final_response)
    except Exception as e:
        print("OpenAI Response Streaming error: " + str(e))
        # return 503
