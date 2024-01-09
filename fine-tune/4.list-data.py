import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

# List up to 10 events from a fine-tuning job
finetunelist = openai.FineTuningJob.list_events(id="ftjob-xxxxxxxx", limit=50)

print(finetunelist)
