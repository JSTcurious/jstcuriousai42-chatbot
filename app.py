import gradio as gr
import torch
from transformers import pipeline

# Use GPU if available
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

# Load a lightweight model that works within Space limits
chatbot = pipeline(
    "text-generation",
#   model="HuggingFaceH4/zephyr-7b-beta",
    model="sshleifer/tiny-gpt2",
    device=device,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
)

# Response function
def respond(message):
    prompt = f"{message}\n"
    output = chatbot(prompt)[0]["generated_text"]
    reply = output.replace(prompt, "").strip() if prompt in output else output.strip()
    return reply

# Gradio UI using Blocks (Gradio 5.x style)
with gr.Blocks(title="JSTcuriousAI42 Chatbot") as demo:
    gr.Markdown("## 🤖 JSTcuriousAI42 Chatbot")
    
    with gr.Row():
        user_input = gr.Textbox(label="Ask something", placeholder="Type your message here...", lines=2)
    
    with gr.Row():
        response = gr.Textbox(label="Response", lines=4)
    
    with gr.Row():
        send_btn = gr.Button("Get Answer")
    
    send_btn.click(fn=respond, inputs=user_input, outputs=response)

demo.launch()

