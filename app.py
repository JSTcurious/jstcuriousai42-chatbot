# Import Gradio for building the web UI
import gradio as gr

# Import the Hugging Face Transformers pipeline
from transformers import pipeline

# Initialize the text generation pipeline using Zephyr-7B-Beta model
# This loads the model and sets up generation parameters
chatbot = pipeline(
    "text-generation",                        # Specify task type
    model="HuggingFaceH4/zephyr-7b-beta",     # You can swap in "mistralai/Mistral-7B-Instruct-v0.2"
    device=0,                                 # Use GPU (device=0); -1 for CPU
    max_new_tokens=256,                       # Limit on response length
    do_sample=True,                           # Enable sampling (more creative responses)
    temperature=0.7                           # Controls randomness (lower = more focused, higher = more diverse)
)

# Function to generate response from the chatbot
def respond(message):
    # Use the message as the prompt
    prompt = f"{message}\n"
    
    # Generate output from the model
    output = chatbot(prompt, do_sample=True, temperature=0.7)[0]["generated_text"]
    
    # Strip the input from the output to return only the model’s reply
    reply = output.replace(prompt, "").strip()
    
    return reply

# Create a Gradio interface to interact with the chatbot
demo = gr.Interface(
    fn=respond,                               # Function to call when user submits input
    inputs=gr.Textbox(label="Ask something"), # Input field for user's question
    outputs=gr.Textbox(label="Response"),     # Output field for model's reply
    title="JSTcuriousAI42 Chatbot"            # Title shown on the UI
)

# Launch the Gradio app (starts the server and opens the web interface)
demo.launch()
