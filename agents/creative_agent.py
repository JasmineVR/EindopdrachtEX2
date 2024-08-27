import openai
import os
import re
import requests
from dotenv import load_dotenv
from PIL import Image
import io

load_dotenv()

def sanitize_prompt(prompt):
    unsafe_keywords = ['violence', 'abuse', 'drugs', 'weapons']  # Add more keywords as needed
    sanitized_prompt = prompt
    for keyword in unsafe_keywords:
        sanitized_prompt = re.sub(rf'\b{keyword}\b', '', sanitized_prompt, flags=re.IGNORECASE)
    sanitized_prompt = re.sub(r'\s+', ' ', sanitized_prompt).strip()  # Clean up extra spaces
    return sanitized_prompt

def generate_art(narrative):
    openai.api_key = os.getenv("API_KEY")
    
    # Add watercolor style to the prompt
    prompt = f"Generate a watercolor painting based on the following description: {narrative}. No text included."
    
    try:
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="512x512"
        )
        
        print("API Response:", response)
        
        if 'data' in response and len(response['data']) > 0:
            image_url = response['data'][0]['url']
            image_data = requests.get(image_url).content
            image = Image.open(io.BytesIO(image_data))
            return image
        else:
            print("Image generation failed, no data returned.")
            return None

    except openai.error.InvalidRequestError as e:
        print(f"InvalidRequestError: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
