import openai
import streamlit as st
from lib import *

# initialized variables state names and values
session_variables = [
    {"name": "submitted", "value": False},
    {"name": "field_textarea_value", "value": ""},
    {"name": "final_response", "value": ""},
]
rag_story = "Anna is a simple country girl, she has a basket of 6 apples, 2 pears and 5 bannanas. Today is wednesday for Anna and she went out for a walk because she was feeling happy and wanted to enjoy the bright sun and the day before it was raining the entire day, so she decided to go out with a basket to the market and buy some fruit. On her way home she ate 2 apples and a bannana."

# method to set the initialised variables to state variables
initialise_variables(session_variables)

custom_messages = prompt_request(rag_story, [])


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


st.title("Simple RAG")
"***You have the following text introduced in the prompt***:"
rag_story
st.button(
    label="Suggestion 1",
    key=1,
    on_click=lambda: on_suggestion_clicked(1),
)
st.button(
    label="Suggestion 2",
    key=2,
    on_click=lambda: on_suggestion_clicked(2),
)
st.session_state.field_textarea_value = st.text_input(
    value=st.session_state.field_textarea_value,
    label="your prompt:",
)
st.button(
    label="Send" if st.session_state.submitted == False else "Clear",
    key=4,
    on_click=lambda: on_run_clicked(),
)
res_container = st.empty()
process_response(completion)
