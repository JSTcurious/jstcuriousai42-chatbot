# JSTcuriousAI42 Chatbot 🤖

This is a lightweight GenAI chatbot powered by HuggingFace's Zephyr-7B, deployed via Gradio.

## 🚀 Features
- Gradio-based web UI
- GPU-accelerated inference (T4)
- Custom prompt/response cleanup

## 🧠 Model
Currently using `HuggingFaceH4/zephyr-7b-beta`. You can swap in `mistralai/Mistral-7B-Instruct-v0.2` or others.

## 🛠️ Setup

```bash
pip install -r requirements.txt
python app.py
