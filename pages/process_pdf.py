import openai
import streamlit as st
from lib import *

# initialized variables state names and values
session_variables = [
    {"name": "submitted", "value": False},
    {"name": "field_textarea_value", "value": ""},
    {"name": "final_response", "value": ""},
]

# method to set the initialised variables to state variables
initialise_variables(session_variables)

custom_messages = initial_object_prompts()


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

st.title("GPT4 PDF processor")
st.session_state.field_textarea_value = st.text_input(
    label="Add .pdf file:",
)
st.button(
    label="Browse" if st.session_state.submitted == False else "Clear",
    key=4,
    on_click=lambda: read_pdf(),
)
res_container = st.empty()
process_response(completion)