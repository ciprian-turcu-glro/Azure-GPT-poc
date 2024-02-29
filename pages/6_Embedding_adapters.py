import streamlit as st
from lib import *


# ----------------
#
# VARIABLES INIT
#
# ----------------

# initialized variables state names and values
session_variables = [
    {"name": "submitted", "value": False},
    {"name": "field_textarea_value", "value": ""},
    {"name": "final_response", "value": ""},
    {
        "name": "rag_options",
        "value": ["Custom", "Option 1", "Option 2", "Option 3", "Option 4"],
    },
]
run_prompt = False
rag_story = ""

# method to set the initialised variables to state variables
initialise_variables(session_variables)

# ----------------
#
# UI
#
# ----------------
st.title("RAG Embedding Adapters")

"Training a model on embeddings and returned results to identify the most relevant results based on trained (labelled) data"

embedding_adapters()