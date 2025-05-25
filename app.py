import gradio as gr
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "mistralai/Mistral-7B-Instruct-v0.3"
hf_token = os.environ.get("HF_TOKEN")  # Needed for gated model access

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=hf_token,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Define response function
def respond(user_message):
    messages = [
        {"role": "system", "content": "You are a helpful, friendly AI assistant created by Jitender to explore Generative AI."},
        {"role": "user", "content": user_message}
    ]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
    outputs = model.generate(inputs, max_new_tokens=256)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split(user_message)[-1].strip()

# Gradio UI
with gr.Blocks(title="JSTcuriousAI42 Chatbot") as demo:
    gr.Markdown("## 🤖 JSTcuriousAI42 Chatbot powered by Mistral 7B v0.3")
    with gr.Row():
        user_input = gr.Textbox(label="Your question", placeholder="Ask me anything...", lines=2)
    with gr.Row():
        response = gr.Textbox(label="Response", lines=6)
    with gr.Row():
        submit_btn = gr.Button("Get Answer")
        submit_btn.click(fn=respond, inputs=user_input, outputs=response)

demo.launch()

