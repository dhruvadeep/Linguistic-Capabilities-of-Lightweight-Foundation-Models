# Linguistic-Capabilities-of-Lightweight-Foundation-Models

## Overview

This project explores the **linguistic capabilities** of **lightweight foundation models**, specifically focusing on tasks like **Named Entity Recognition (NER)** and **Part-of-Speech (POS) Tagging**.

We fine-tune small, efficient foundation models using Hugging Face libraries and optimized training strategies (like PEFT and Flash Attention) to adapt them for linguistic tasks.

---

## Installation

First, install the required libraries:

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install peft wandb dataloader datasets huggingface_hub trl flash_attn bitsandbytes
pip install -U accelerate
```

Then, navigate to the dataset folder:

```bash
cd ./Dataset
```

Install additional dependencies:

```bash
pip install pandas datasets
```

Download and preprocess the dataset:

```bash
python3 dataset.py
```

This will **download** the dataset from the internet and **convert** it into JSON format for training.

---

## Setting Up

In each of the `.ipynb` files (like `train.ipynb`), you need to **login** to Hugging Face Hub by adding your token where indicated:

```python
from huggingface_hub import login
login(token="your-huggingface-token-here")
```

---

## Model Training

Currently, the provided training example (`train.ipynb`) is designed for **NER tagging**.

If you want to modify it for **POS tagging** instead:

1. **Change the system prompt**:

**Original (NER Prompt):**

```python
system_prompt = """
You are a model which is optimized for NER Tagging. The input you receive should be processed word by word, returning the NER tag for each word...
"""
```

**New (POS Prompt Example):**

```python
system_prompt = """
You are a model optimized for Part-of-Speech (POS) tagging. For each input sentence, output a POS tag per word, using standard POS tags like 'NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'CONJ', etc.
"""
```

2. **Change the model names**:

**Original:**

```python
base_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
new_model = "deepseek_r1_1.5b_ner"
```

**For POS tagging:**

```python
base_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
new_model = "deepseek_r1_1.5b_pos"
```

Make sure you update both the `base_model` and `new_model` names accordingly in your training script.

---

## Saving the Model

After training by running all the cells in `train.ipynb`, you **must** run all the cells in `save_model_weights.ipynb` to properly save your fine-tuned model.

---

## Testing the Fine-Tuned Model

To test your newly trained model, you can use:

```python
input_text = """Please return the NER tags for each word for 'Peter, pass the butter, please'"""
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
```

Make sure to change the prompt and model call appropriately if you adapted the training for **POS tagging** or any other task.

> :warning: **Important Warning**: After saving your model, navigate to the folder where the model is saved and **delete any checkpoint folders or unrelated files** before using or uploading the model. Keeping unnecessary files might cause loading issues or extra storage usage.

---

## Notes

- Always make sure your Hugging Face token has permission to download models and upload fine-tuned models if needed.
- The training configurations (batch size, learning rate, etc.) can be adjusted inside `train.ipynb` depending on your hardware capabilities.
- Dataset conversion and preprocessing must be completed **before** training.

---

## License

This project is intended for research and educational purposes.
