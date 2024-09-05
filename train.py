from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch
import os
import math
from transformers import AdamW
from torch.optim.lr_scheduler import LambdaLR

import process_text, randmnize_null_gpt2 
from datasets import load_dataset

process_text.clean("GPT2-Architecture-Bible-main\Bible_KJV.txt")

model,tokenizer = randmnize_null_gpt2.randomize("GPT2-Architecture-Bible-main/NullGPT2")

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
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    fp16=True,
)
def custom_lr_scheduler(current_step, total_steps, initial_lr):
    return initial_lr * math.log(-current_step + total_steps + 1, total_steps)

class CustomLRScheduler(LambdaLR):
    def __init__(self, optimizer, total_steps, initial_lr, last_epoch=-1):
        self.total_steps = total_steps
        self.initial_lr = initial_lr
        super().__init__(optimizer, lr_lambda=self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, current_step):
        return custom_lr_scheduler(current_step, self.total_steps, self.initial_lr)

# Example usage

optimizer = AdamW(model.parameters(), lr=8e-5)
total_steps = len(tokenized_datasets['train']) // training_args.per_device_train_batch_size * training_args.num_train_epochs
scheduler = CustomLRScheduler(optimizer, total_steps, initial_lr=8e-5)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    data_collator=data_collator,
    optimizers=(optimizer, scheduler)
)


# Move model to GPU
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

trainer.train()

# Save the model
output_dir = "./finetuned_gpt2_large"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model saved to {output_dir}")
