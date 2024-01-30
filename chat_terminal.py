import openai
import streamlit as st
from lib import *

# initialized variables state names and values
input_text = ""
while True:
    user_input = input("Enter a prompt (type 'quit' to exit): ")
    custom_messages = custom_messages_generating(user_input)
    if user_input.lower() == "quit":
        print("Exiting the app...")
        break
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
    process_terminal_response(completion)
