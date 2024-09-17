import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

# Load the dataset
with open('nlq_dataset.json', 'r') as f:
    data = json.load(f)

# Prepare the dataset
def prepare_data(examples):
    return {
        "text": f"Natural Query: {examples['natural_query']}\nSQL Query: {examples['sql_query']}"
    }

dataset = Dataset.from_dict({
    "natural_query": [item["natural_query"] for item in data],
    "sql_query": [item["sql_query"] for item in data]
})
dataset = dataset.map(prepare_data, remove_columns=dataset.column_names)

# Split the dataset
train_val = dataset.train_test_split(test_size=0.1)
train_data = train_val['train']
val_data = train_val['test']

# Load the model and tokenizer
model_name = "meta-llama/Meta-Llama-3.1-8B"  # Make sure you have access to this model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# Quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load the model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

# Prepare the model for k-bit training
model = prepare_model_for_kbit_training(model)

# LoRA configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA to the model
model = get_peft_model(model, peft_config)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

train_dataset = train_data.map(tokenize_function, batched=True, remove_columns=["text"])
val_dataset = val_data.map(tokenize_function, batched=True, remove_columns=["text"])

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    save_total_limit=2,
    learning_rate=2e-5,
    fp16=True,
    gradient_accumulation_steps=4,
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# Start training
trainer.train()

# Save the fine-tuned model
trainer.save_model("./fine_tuned_llama3")