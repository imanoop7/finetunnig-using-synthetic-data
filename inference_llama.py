import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import gradio as gr

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

# Gradio interface
def gradio_interface(natural_query):
    sql_query = generate_sql_query(natural_query)
    return sql_query

iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(lines=2, placeholder="Enter your natural language query here..."),
    outputs=gr.Textbox(label="Generated SQL Query"),
    title="Natural Language to SQL Query Generator",
    description="Enter a natural language query, and the model will generate a corresponding SQL query.",
    examples=[
        ["What are the top 5 products by sales in the Electronics category?"],
        ["Show me all customers who made a purchase in the last 30 days."],
        ["What is the average order value for each country?"]
    ]
)

if __name__ == "__main__":
    iface.launch()