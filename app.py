import gradio as gr
import torch
from transformers import pipeline

# Smart device setting: use GPU if available, else CPU
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

# Load the model pipeline
chatbot = pipeline(
    "text-generation",
    model="HuggingFaceH4/zephyr-7b-beta",
    device=device,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
)

# Chat function
def respond(message):
    prompt = f"{message}\n"
    output = chatbot(prompt)[0]["generated_text"]
    reply = output.replace(prompt, "").strip() if prompt in output else output.strip()
    return reply

# Gradio UI
demo = gr.Interface(
    fn=respond,
    inputs=gr.Textbox(label="Ask something"),
    outputs=gr.Textbox(label="Response"),
    title="JSTcuriousAI42 Chatbot"
)

# Launch app
demo.launch()
