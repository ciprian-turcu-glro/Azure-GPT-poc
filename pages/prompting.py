import openai
import streamlit as st
from lib import *

# initialized variables state names and values
session_variables = [
    {"name": "submitted", "value": False},
    {"name": "field_textarea_value", "value": ""},
    {"name": "final_response", "value": ""},
    {"name": "rag_selectbox", "value": ""},
    {
        "name": "rag_options",
        "value": ["Custom", "Suggestion 1", "Suggestion 2", "Suggestion 3"],
    },
]
rag_story = "Anna is a simple country girl, she has a basket of 6 apples, 2 pears and 5 bananas. Today is wednesday for Anna and she went out for a walk because she was feeling happy and wanted to enjoy the bright sun and the day before it was raining the entire day, so she decided to go out with a basket to the market and buy some fruit. On her way home she ate 2 apples and a banana."

# method to set the initialised variables to state variables
initialise_variables(session_variables)


st.title("Simple Context Prompting")
"***You have the following text introduced in the prompt***:"
rag_story

rag_option = st.selectbox(
    "Automatic sugestions:",
    (st.session_state.rag_options),
    key=0,
)
prompt_value = generate_custom_text(rag_option)
prompt_value = st.text_input(
    key="field_prompt_value",
    value=prompt_value,
    label="your prompt:",
)

custom_messages = prompt_request(rag_story, [], prompt=prompt_value)
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
print(
    "1------------------------------------------------------------------------******************************************************"
)
print(
    "1------------------------------------------------------------------------******************************************************"
)
st.button(
    label="Send" if st.session_state.submitted == False else "Clear",
    key=4,
    on_click=lambda: on_run_clicked(),
)
res_container = st.empty()
process_response(completion)
