---
title: JSTcurious Chatbot
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.31.0
app_file: app.py
pinned: true
---

# 🤖 JSTcurious Chatbot powered by Mistral 7B v0.3

[![Built with Gradio](https://img.shields.io/badge/Built%20with-Gradio-blue)](https://gradio.app)
[![Model: Mistral-7B-Instruct-v0.3](https://img.shields.io/badge/Mistral-7B--Instruct--v0.3-red)](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)

This is an interactive chatbot demo powered by the `mistralai/Mistral-7B-Instruct-v0.3` model, using Hugging Face Transformers and Gradio. It features token streaming, formatted responses, and runs efficiently on GPU (T4 or A10G recommended).

---

## ✨ Features

- Streaming output (token-by-token generation)
- Markdown rendering for formatting and newlines
- Fast inference with GPU acceleration
- Runs entirely on Hugging Face Spaces

---

## 📦 Dependencies

This app requires the following environment setup in `requirements.txt`:

```txt
gradio==5.31.0                  # UI framework with streaming support
transformers==4.42.2            # Required for Mistral v0.3 + chat_template + streaming
torch==2.2.2                    # GPU-compatible for inference
accelerate                      # Optional, enables device_map="auto"
numpy<2                         # Prevents known PyTorch compatibility issues
sentencepiece                   # Required for tokenizer loading (SentencePiece-based)

