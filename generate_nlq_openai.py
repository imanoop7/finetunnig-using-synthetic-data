import json
import time
from tqdm import tqdm
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Set your OpenAI API key in the environment variable OPENAI_API_KEY

def generate_nlq_pair():
    llm = OpenAI(temperature=0.7, max_tokens=500)

    response_schemas = [
        ResponseSchema(name="natural_query", description="The natural language query"),
        ResponseSchema(name="sql_query", description="The corresponding SQL query"),
        ResponseSchema(name="tables", description="A list of tables involved in the query"),
        ResponseSchema(name="complexity", description="A rating of query complexity (1-5)")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    format_instructions = output_parser.get_format_instructions()

    prompt = PromptTemplate(
        template="Generate a complex and realistic natural language query and its corresponding SQL query for a hypothetical database. The query should involve multiple tables and operations.\n{format_instructions}\n\nExample:\n{example}\n\nGenerate a different, equally complex query:",
        input_variables=[],
        partial_variables={"format_instructions": format_instructions, "example": json.dumps({
            "natural_query": "What are the top 5 products by revenue in the Electronics category for customers in New York during the last quarter, excluding weekends?",
            "sql_query": "SELECT p.product_name, SUM(o.quantity * p.price) as revenue FROM products p JOIN order_items o ON p.product_id = o.product_id JOIN orders ord ON o.order_id = ord.order_id JOIN customers c ON ord.customer_id = c.customer_id WHERE p.category = 'Electronics' AND c.state = 'New York' AND ord.order_date >= DATE_SUB(CURDATE(), INTERVAL 3 MONTH) AND DAYOFWEEK(ord.order_date) NOT IN (1, 7) GROUP BY p.product_id ORDER BY revenue DESC LIMIT 5",
            "tables": ["products", "order_items", "orders", "customers"],
            "complexity": 4
        }, indent=2)}
    )

    try:
        output = llm(prompt.format())
        return output_parser.parse(output)
    except Exception as e:
        print(f"Error generating pair: {e}")
        return None

# Generate 100 NLQ pairs
dataset = []
for _ in tqdm(range(100), desc="Generating NLQ pairs"):
    pair = generate_nlq_pair()
    if pair:
        dataset.append(pair)
    time.sleep(1)  # To avoid hitting API rate limits

# Save the dataset
with open('nlq_dataset_openai_langchain.json', 'w') as f:
    json.dump(dataset, f, indent=2)

print(f"Generated {len(dataset)} NLQ pairs and saved to nlq_dataset_openai_langchain.json")