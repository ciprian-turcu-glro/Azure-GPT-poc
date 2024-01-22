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

st.session_state.pdf_text = ""

st.title("GPT4 PDF processor")
st.session_state.field_textarea_value = st.text_input(
    label="Pass .PDF file path:",
)

st.button(
    label="Load" if st.session_state.submitted == False else "Clear",
    key=4,
    on_click=lambda: read_pdf(st.session_state.field_textarea_value),
)

st.text_area("LOADED .PDF FILE", st.session_state.pdf_text)
