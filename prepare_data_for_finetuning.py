import json
from sklearn.model_selection import train_test_split

# Load the dataset
with open('nlq_dataset.json', 'r') as f:
    dataset = json.load(f)

# Split into train and validation sets
train_data, val_data = train_test_split(dataset, test_size=0.1, random_state=42)

# Format data for fine-tuning
def format_for_training(data):
    return [
        f"Natural Query: {item['natural_query']}\nSQL Query: {item['sql_query']}"
        for item in data
    ]

train_texts = format_for_training(train_data)
val_texts = format_for_training(val_data)

# Save formatted data
with open('train_data.txt', 'w') as f:
    f.write('\n\n'.join(train_texts))

with open('val_data.txt', 'w') as f:
    f.write('\n\n'.join(val_texts))

print(f"Prepared {len(train_texts)} training samples and {len(val_texts)} validation samples")