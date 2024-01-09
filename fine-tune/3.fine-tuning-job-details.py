# get finetuned model details
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
fine_tuning_job_ID = "ftjob-xxxxxx"
# Retrieve the state of a fine-tune
job = openai.FineTuningJob.retrieve(fine_tuning_job_ID)
print(job)
