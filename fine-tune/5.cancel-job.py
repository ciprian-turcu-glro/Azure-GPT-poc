# Cancel a job
import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

result = openai.FineTuningJob.cancel("ftjob-xxxxxxxx")

print(result)