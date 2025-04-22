# -*- coding: utf-8 -*-

pip install transformers torch datasets

pip install wandb

pip install transformers torch datasets scikit-learn tensorboard

# Import necessary libraries
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Step 1: Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Step 2: Load and prepare the dataset
dataset = load_dataset("imdb")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(200))
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Step 3: Define evaluation metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Step 4: Set training arguments with TensorBoard logging
training_args = TrainingArguments(
    output_dir="./results",          # Directory for model checkpoints
    num_train_epochs=3,             # Train for 3 epochs
    per_device_train_batch_size=8,  # Batch size for training
    per_device_eval_batch_size=8,   # Batch size for evaluation
    warmup_steps=500,               # Warmup steps for learning rate
    weight_decay=0.01,             # Regularization
    logging_dir="./logs",           # Directory for TensorBoard logs
    logging_steps=10,              # Log every 10 steps
    evaluation_strategy="epoch",    # Evaluate after each epoch
    save_strategy="epoch",         # Save after each epoch
    load_best_model_at_end=True,    # Load best model at the end
    report_to="tensorboard",        # Use TensorBoard for logging (no API key needed)
)

# Step 5: Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Step 6: Fine-tune the model
trainer.train()

# Step 7: Evaluate the model
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)

# Step 8: Save the fine-tuned model
model.save_pretrained("./fine_tuned_bert")
tokenizer.save_pretrained("./fine_tuned_bert")

# Step 9: Inference example
fine_tuned_model = BertForSequenceClassification.from_pretrained("./fine_tuned_bert")
fine_tuned_tokenizer = BertTokenizer.from_pretrained("./fine_tuned_bert")
text = "This movie was amazing!"
inputs = fine_tuned_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
with torch.no_grad():
    outputs = fine_tuned_model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
labels = ["negative", "positive"]
print(f"Prediction for '{text}': {labels[prediction]}")

