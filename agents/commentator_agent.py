import openai
import os
from dotenv import load_dotenv  # Ensure dotenv is imported

# Load the .env file
load_dotenv()

def commentator_agent(behavior_description):
    openai.api_key = os.getenv("API_KEY")

    if not openai.api_key:
        raise ValueError("API key not found. Please set it in the .env file.")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that provides detailed commentary."},
            {"role": "user", "content": f"Analyze the following behavior: {behavior_description}. Provide a detailed commentary."}
        ],
        max_tokens=150
    )

    commentary = response.choices[0].message['content'].strip()
    return commentary
