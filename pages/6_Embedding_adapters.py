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
st.title("RAG Embedding Adapters")

"Training a model on embeddings and returned results to identify the most relevant results based on trained (labelled) data"

embedding_adapters()