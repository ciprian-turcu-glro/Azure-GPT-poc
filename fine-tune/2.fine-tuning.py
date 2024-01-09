# The actual fine tuning of the model, based on the uploaded file
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

training_file_ID = "file-xxxxxxx"
model_to_finetune = "gpt-3.5-turbo"
job = openai.FineTuningJob.create(
    training_file=training_file_ID,
    model=model_to_finetune,
    hyperparameters={"n_epochs": 3},
)
print(job)
