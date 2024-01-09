import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
fileID = openai.File.create(
  file=open("./fine-tune/train_dataset.jsonl", "rb"),
  purpose='fine-tune'
)
# prints out the file upload details to be used for fine tuning later
print(fileID)
