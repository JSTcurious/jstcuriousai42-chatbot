import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import gradio as gr
import threading
import os

model_id = "mistralai/Mistral-7B-Instruct-v0.3"
hf_token = os.environ.get("HF_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    token=hf_token
)

def generate_response_stream(prompt):
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {
        "inputs": inputs,
        "streamer": streamer,
        "max_new_tokens": 128,
        "temperature": 0.7,
        "do_sample": False
    }

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    response = ""
    for token in streamer:
        response += token
        yield response.replace("\n", "<br>")

with gr.Blocks(title="JSTcurious Chatbot") as demo:
    gr.Markdown("## 🤖 JSTcurious Chatbot powered by Mistral 7B v0.3")
    prompt = gr.Textbox(label="Your question", placeholder="Ask me anything...")
    response = gr.Markdown(label="Response", show_label=True)
    submit = gr.Button("Get Answer")

    submit.click(fn=generate_response_stream, inputs=prompt, outputs=response, show_progress=True)

demo.launch()

