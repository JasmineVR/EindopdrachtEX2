import os
import cv2
import numpy as np
from dotenv import load_dotenv
from PIL import Image
import io
import datetime
import requests
import openai
import matplotlib.pyplot as plt
from agents.vision_agent import load_model, vision_agent
from agents.commentator_agent import commentator_agent
from agents.knowledge_aggregator_agent import knowledge_aggregator
from agents.creative_agent import generate_art

# Load the .env file
load_dotenv()

# Set the API key
openai.api_key = os.getenv("API_KEY")

def run_pipeline(frame, model):
    # Vision Agent
    detected_behaviors = vision_agent(frame, model)
    behavior_description = ", ".join(detected_behaviors)
    print(f"Detected Behaviors: {behavior_description}")
    
    # Commentator Agent
    commentary = commentator_agent(behavior_description)
    print(f"Commentary: {commentary}")
    
    # Knowledge Aggregator Agent
    narrative = knowledge_aggregator(behavior_description, commentary)
    print(f"Narrative: {narrative}")
    
    # Creative Agent
    final_artwork = generate_art(narrative)
    
    return final_artwork, detected_behaviors

def display_image(image, title="Generated Artwork"):
    """Display an image using PIL."""
    plt.figure()
    plt.imshow(image)
    plt.axis('off')  # Hide axes
    plt.title(title)
    plt.show()

def display_artwork(image):
    """Display an image using PIL."""
    plt.figure()
    plt.imshow(image)
    plt.axis('off')  # Hide axes
    plt.title("Generated Artwork")
    plt.show()

def display_camera_feed(frame):
    """Display the webcam feed."""
    cv2.imshow('Camera Feed', frame)

if __name__ == "__main__":
    model = load_model()
    cap = cv2.VideoCapture(0)  # Use 0 for default webcam
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        # Run the pipeline
        generated_image, detected_behaviors = run_pipeline(frame, model)
        
        if generated_image:
            # Save the generated image to the output_images directory with a unique filename
            output_dir = "output_images"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"generated_artwork_{timestamp}.png")
            generated_image.save(output_path)
            print(f"Artwork saved to {output_path}")
            
            # Display the generated image
            display_artwork(generated_image)
        else:
            print("No image generated due to an issue with the prompt.")
        
        # Display the frame with detected behaviors
        cv2.putText(frame, f"Detected Behaviors: {', '.join(detected_behaviors)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        display_camera_feed(frame)
        
        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
