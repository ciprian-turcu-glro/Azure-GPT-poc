import openai
import streamlit as st
from lib import *

# ----------------
#
# VARIABLES INIT
#
# ----------------

# initialized variables state names and values
session_variables = get_session_variables()

run_prompt = False
rag_story = ""

# method to set the initialised variables to state variables
initialise_variables(session_variables)

# ----------------
#
# UI
#
# ----------------
st.title("RAG Query expansion with generated answers")

"Expansion with generated answers if basically prompting the LLM with the original prompt + an ask to generate x(5) similar questions and using it's answer, prefixing it to the original prompt when doing retrieval with Chroma"

rag_option = st.selectbox(
    "Automatic sugestions:",
    (st.session_state.rag_options),
    key=0,
)
prompt_value = generate_custom_text_for_simple_query_expansion(rag_option)
prompt_value = st.text_input(
    key="field_prompt_value",
    value=prompt_value,
    label="prompt:",
)
# ----------------
#
# REQUEST AND RAG FUNCTIONALITY
#
# ----------------
if st.session_state.submitted:
    # first request to the LLM directly
    messages = [
        {
            "role": "system",
            "content": "You are a helpful expert financial research assistant. Your users are asking questions about information contained in an annual report."
            "Suggest up to 5 additional related questions to help them find the information they need, for the provided question"
            "Suggest only short questions without compound sentances. Suggest a variety of questions that cover different aspects of the topic."
            "Make sure they are complete questions, and they are related to the original question.",
        },
        {
            "role": "user",
            "content": f"Question: {prompt_value}",
        },
    ]
    augmented_response = openai_prompt_request(prompt_value, custom_messages=messages)
    augmented_prompt = augmented_response + prompt_value
    # generate propper request for
    custom_messages = custom_messages_generating(rag_story, [], prompt=augmented_prompt)
    completion = openai_prompt_request(
        augmented_prompt, "Chip-GPT4-32k", custom_messages
    )
    augmented_prompted_response = completion + prompt_value
    retrieved_documents = apply_rag(query=augmented_prompted_response)
    messages = [
        {
            "role": "system",
             "content": "You are a helpful expert financial research assistant. Your users are asking questions about information contained in an annual report."
            "You will be shown the user's question, and the relevant information from the annual report. Answer the user's question using only this information.",
        },
        {
            "role": "user",
            "content": f"Question: {completion+prompt_value}. \n Information: {retrieved_documents}",
        },
    ]

    completion = openai.ChatCompletion.create(
        engine="Chip-GPT4-32k",
        messages=messages,
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
