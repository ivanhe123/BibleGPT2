from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch
import os
import process_text, randmnize_null_gpt2 
from datasets import load_dataset

process_text.clean("cleared.txt")

model,tokenizer = randmnize_null_gpt2.randomize("Inoob/NullGPT2")

# Set the padding token
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset('text', data_files={'train': 'cleared.txt'})
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Create a data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    data_collator=data_collator,
)

# Move model to GPU
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

trainer.train()

# Save the model
output_dir = "./finetuned_gpt2_large"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model saved to {output_dir}")
