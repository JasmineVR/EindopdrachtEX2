import openai
import os
from dotenv import load_dotenv

load_dotenv()

def knowledge_aggregator(detected_behaviors, commentary):
    openai.api_key = os.getenv("API_KEY")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that combines behaviors and commentary into a narrative."},
            {"role": "user", "content": f"Combine the following detected behaviors: {detected_behaviors} with this commentary: {commentary}. Form a coherent narrative."}
        ],
        max_tokens=150
    )

    narrative = response['choices'][0]['message']['content'].strip()
    return narrative
