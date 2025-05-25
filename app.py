import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Model name
model_id = "mistralai/Mistral-7B-Instruct-v0.2"

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    device_map="auto"  # automatically selects the GPU or CPU
)

# Create the pipeline
chatbot = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7
)

# Prompt handler
def respond(message):
    system_prompt = "You are a helpful, friendly AI assistant created by Jitender to explore Generative AI."
    full_prompt = f"<s>[INST] {system_prompt}\n{message} [/INST]"
    output = chatbot(full_prompt)[0]["generated_text"]
    return output.split("[/INST]")[-1].strip()

# Gradio UI
with gr.Blocks(title="JSTcuriousAI42 Chatbot") as demo:
    gr.Markdown("## 🤖 JSTcuriousAI42 Chatbot powered by Mistral")
    user_input = gr.Textbox(label="Your question", lines=2)
    response = gr.Textbox(label="Response", lines=4)
    button = gr.Button("Get Answer")
    button.click(fn=respond, inputs=user_input, outputs=response)

demo.launch()

