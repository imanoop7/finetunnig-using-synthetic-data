import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from sklearn.metrics import accuracy_score
import numpy as np

# Load the fine-tuned model and tokenizer
model_path = "./fine_tuned_llama3"
peft_config = PeftConfig.from_pretrained(model_path)
model_name = peft_config.base_model_name_or_path

# Quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

# Load the fine-tuned model
model = PeftModel.from_pretrained(base_model, model_path)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load the test dataset
with open('nlq_dataset.json', 'r') as f:
    data = json.load(f)

# Use a subset of the data for evaluation (e.g., 100 samples)
test_data = data[:100]

def generate_sql_query(natural_query):
    prompt = f"Natural Query: {natural_query}\nSQL Query:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sql_query = generated_text.split("SQL Query:")[1].strip()
    return sql_query

def calculate_bleu(reference, candidate):
    return sentence_bleu([reference.split()], candidate.split())

def calculate_rouge(reference, candidate):
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference)
    return scores[0]['rouge-l']['f']

def exact_match(reference, candidate):
    return int(reference.lower() == candidate.lower())

# Evaluate the model
bleu_scores = []
rouge_scores = []
exact_matches = []

for sample in tqdm(test_data, desc="Evaluating"):
    natural_query = sample['natural_query']
    reference_sql = sample['sql_query']
    
    generated_sql = generate_sql_query(natural_query)
    
    bleu_scores.append(calculate_bleu(reference_sql, generated_sql))
    rouge_scores.append(calculate_rouge(reference_sql, generated_sql))
    exact_matches.append(exact_match(reference_sql, generated_sql))

# Calculate average scores
avg_bleu = np.mean(bleu_scores)
avg_rouge = np.mean(rouge_scores)
exact_match_accuracy = np.mean(exact_matches)

print(f"Average BLEU score: {avg_bleu:.4f}")
print(f"Average ROUGE-L score: {avg_rouge:.4f}")
print(f"Exact match accuracy: {exact_match_accuracy:.4f}")

# Save detailed results
results = {
    "bleu_scores": bleu_scores,
    "rouge_scores": rouge_scores,
    "exact_matches": exact_matches,
    "avg_bleu": avg_bleu,
    "avg_rouge": avg_rouge,
    "exact_match_accuracy": exact_match_accuracy
}

with open('evaluation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Detailed results saved to evaluation_results.json")