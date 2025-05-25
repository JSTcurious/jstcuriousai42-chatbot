import gradio as gr
import torch
from transformers import pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

chatbot = pipeline(
    "text-generation",
    model="sshleifer/tiny-gpt2",  # small model to stay within memory limits
    device=0 if device.type == "cuda" else -1,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
)

def respond(message):
    prompt = f"{message}\n"
    output = chatbot(prompt)[0]["generated_text"]
    reply = output.replace(prompt, "").strip() if prompt in output else output.strip()
    return reply

with gr.Blocks(title="JSTcuriousAI42 Chatbot") as demo:
    gr.Markdown("## 🤖 JSTcuriousAI42 Chatbot")
    user_input = gr.Textbox(label="Ask something", lines=2)
    response_output = gr.Textbox(label="Response", lines=4)
    send_btn = gr.Button("Get Answer")
    send_btn.click(fn=respond, inputs=user_input, outputs=response_output)

demo.launch()

