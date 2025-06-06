# -*- coding: utf-8 -*-


pip install bert-extractive-summarizer

from summarizer import Summarizer
text = open("article.txt").read()
model = Summarizer('distilbert-base-uncased')
result = model(text, num_sentences=3)
print(result)

"""**Text Generator**"""

import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
import numpy as np

# Load pre-trained BERT model and tokenizer from Hugging Face
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Hyperparameters
MAX_LEN = 20  # Max sequence length for input and output
VOCAB_SIZE = tokenizer.vocab_size  # ~30,522 for BERT base
EMBEDDING_DIM = 768  # BERT's hidden size

# Define the generative model
def build_generator():
    input_ids = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.layers.Input(shape=(MAX_LEN,), dtype=tf.int32, name="attention_mask")

    # Get BERT embeddings
    bert_outputs = bert_model(input_ids, attention_mask=attention_mask)[0]  # [batch, seq_len, 768]

    # Add a dense layer for token prediction
    outputs = tf.keras.layers.Dense(VOCAB_SIZE, activation='softmax')(bert_outputs)

    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=outputs)
    return model

# Build and compile the model
generator = build_generator()
generator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy')

# Preprocess input prompt
def preprocess_prompt(prompt, max_len=MAX_LEN):
    encoding = tokenizer(prompt, max_length=max_len, padding='max_length',
                         truncation=True, return_tensors='tf')
    return encoding['input_ids'], encoding['attention_mask']

# Generate text
def generate_text(prompt, max_tokens=10, temperature=1.0):
    input_ids, attention_mask = preprocess_prompt(prompt)
    generated_ids = input_ids.numpy().tolist()[0]  # Start with input_ids

    for _ in range(max_tokens):
        # Predict next token probabilities
        preds = generator.predict([input_ids, attention_mask], verbose=0)  # [1, MAX_LEN, VOCAB_SIZE]

        # Use the last valid position if sequence exceeds MAX_LEN
        position = min(len(generated_ids) - 1, MAX_LEN - 1)
        next_token_logits = preds[0, position, :] / temperature  # Apply temperature

        # Sample the next token
        next_token = tf.random.categorical(tf.expand_dims(next_token_logits, 0), 1)[0, 0].numpy()

        # Append the predicted token
        generated_ids.append(next_token)

        # Update input_ids and attention_mask (truncate to MAX_LEN)
        input_ids = tf.constant([generated_ids[-MAX_LEN:]], dtype=tf.int32)  # Keep last MAX_LEN tokens
        attention_mask = tf.ones_like(input_ids, dtype=tf.int32)

        # Stop if end token ([SEP]) is generated
        if next_token == tokenizer.sep_token_id:
            break

    # Decode the generated token IDs to text
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text

# Example usage
prompt = "The future of AI is"
generated = generate_text(prompt, max_tokens=10, temperature=0.7)
print(f"Generated text: {generated}")

!pip install transformers torch datasets evaluate

# DISABLE W&B LOGGING
import os
os.environ["WANDB_DISABLED"] = "true"

# Now run your BERT fine-tuning code
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)
from datasets import load_dataset
import evaluate
import numpy as np

# Load dataset
dataset = load_dataset("imdb")

# Take smaller subsets for faster training
train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))
eval_dataset = dataset["test"].shuffle(seed=42).select(range(200))

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

# Load model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Set up evaluation
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Configure training (WITHOUT W&B)
training_args = TrainingArguments(
    output_dir="./bert_imdb_results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir="./logs",
    logging_steps=10,
    report_to="none"  # Explicitly tell Trainer not to use W&B
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    compute_metrics=compute_metrics,
)

# Train without W&B
trainer.train()

# Save model
model.save_pretrained("./bert_imdb_model")
tokenizer.save_pretrained("./bert_imdb_model")

# Test predictions
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
print(classifier("This movie was great!"))
