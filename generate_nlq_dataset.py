import json
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker()

def generate_company():
    return {
        "name": fake.company(),
        "sector": fake.company_suffix(),  # Changed from industry to sector
        "founded_year": fake.year()
    }

def generate_product():
    return {
        "name": fake.word(),  # Changed from product_name() to word()
        "category": fake.word(),  # Changed from category() to word()
        "price": round(random.uniform(10, 1000), 2),
        "stock_quantity": random.randint(0, 1000)
    }

def generate_customer():
    return {
        "name": fake.name(),
        "age": random.randint(18, 80),
        "country": fake.country()
    }

def generate_nlq_pair():
    company = generate_company()
    product = generate_product()
    customer = generate_customer()
    
    query_types = [
        "top_n",
        "filter",
        "aggregate",
        "join",
        "date_range",
        "group_by",
        "order_by",
        "complex"
    ]
    query_type = random.choice(query_types)

    if query_type == "top_n":
        n = random.randint(3, 10)
        metric = random.choice(["sales", "revenue", "customers", "orders"])
        period = random.choice(["this month", "this year", "last quarter", "all time"])
        natural_query = f"What are the top {n} products by {metric} for {company['name']} {period}?"
        sql_query = f"SELECT product_name, SUM({metric}) as total_{metric} FROM sales WHERE company_name = '{company['name']}' AND period = '{period}' GROUP BY product_name ORDER BY total_{metric} DESC LIMIT {n}"

    elif query_type == "filter":
        natural_query = f"Show me all products in the {product['category']} category with a price over ${product['price']}."
        sql_query = f"SELECT * FROM products WHERE category = '{product['category']}' AND price > {product['price']}"

    elif query_type == "aggregate":
        function = random.choice(["average", "total", "maximum", "minimum"])
        metric = random.choice(["price", "quantity", "rating"])
        natural_query = f"What is the {function} {metric} of all products for {company['name']}?"
        sql_query = f"SELECT {function.upper()}({metric}) as result FROM products WHERE company_name = '{company['name']}'"

    elif query_type == "join":
        natural_query = f"List all customers and their total order value for {company['name']}."
        sql_query = f"SELECT c.customer_name, SUM(o.order_total) as total_value FROM customers c JOIN orders o ON c.customer_id = o.customer_id WHERE c.company_name = '{company['name']}' GROUP BY c.customer_id"

    elif query_type == "date_range":
        start_date = fake.date_between(start_date='-1y', end_date='today')
        end_date = fake.date_between(start_date=start_date, end_date='today')
        natural_query = f"How many orders were placed for {company['name']} between {start_date} and {end_date}?"
        sql_query = f"SELECT COUNT(*) as order_count FROM orders WHERE company_name = '{company['name']}' AND order_date BETWEEN '{start_date}' AND '{end_date}'"

    elif query_type == "group_by":
        dimension = random.choice(["category", "supplier", "country"])
        metric = random.choice(["sales", "profit", "quantity"])
        natural_query = f"What is the total {metric} for each {dimension} in {company['name']}?"
        sql_query = f"SELECT {dimension}, SUM({metric}) as total_{metric} FROM sales WHERE company_name = '{company['name']}' GROUP BY {dimension} ORDER BY total_{metric} DESC"

    elif query_type == "order_by":
        metric = random.choice(["price", "stock_quantity", "rating"])
        order = random.choice(["highest", "lowest"])
        natural_query = f"List all products of {company['name']} ordered by {metric} from {order} to {'lowest' if order == 'highest' else 'highest'}."
        sql_query = f"SELECT * FROM products WHERE company_name = '{company['name']}' ORDER BY {metric} {'DESC' if order == 'highest' else 'ASC'}"

    else:  # complex
        natural_query = f"What are the top 5 categories by sales for customers aged 25-35 in {company['name']} for the last quarter?"
        sql_query = f"""
        SELECT p.category, SUM(s.sales) as total_sales
        FROM sales s
        JOIN products p ON s.product_id = p.product_id
        JOIN customers c ON s.customer_id = c.customer_id
        WHERE c.age BETWEEN 25 AND 35
        AND s.company_name = '{company['name']}'
        AND s.sale_date >= DATE_SUB(CURDATE(), INTERVAL 3 MONTH)
        GROUP BY p.category
        ORDER BY total_sales DESC
        LIMIT 5
        """

    return {
        "natural_query": natural_query,
        "sql_query": sql_query.strip(),
        "company": company,
        "product": product,
        "customer": customer
    }

# Generate 1000 NLQ pairs
dataset = [generate_nlq_pair() for _ in range(1000)]

# Save the dataset
with open('nlq_dataset.json', 'w') as f:
    json.dump(dataset, f, indent=2)

print(f"Generated {len(dataset)} NLQ pairs and saved to nlq_dataset.json")