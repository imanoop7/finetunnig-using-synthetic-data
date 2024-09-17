# Natural Language to SQL Query Generator

This project implements a system for generating SQL queries from natural language questions using a fine-tuned LLaMA model. It includes scripts for data generation, model fine-tuning, and inference with a user-friendly Gradio interface.

## Project Structure

- `generate_nlq_dataset.py`: Generates synthetic Natural Language Query (NLQ) dataset using Faker.
- `generate_nlq_openai.py`: Generates NLQ dataset using OpenAI's GPT model and Langchain.
- `prepare_data_for_finetuning.py`: Prepares the generated data for fine-tuning.
- `finetune_llama.py`: Fine-tunes the LLaMA model on the prepared dataset.
- `inference_llama.py`: Provides inference capabilities with a Gradio interface.
- `requirements.txt`: Lists all the required Python packages.

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/imanoop7/finetunnig-using-synthetic-data
   cd finetunnig-using-synthetic-data
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key as an environment variable (if using `generate_nlq_openai.py`):
   ```
   export OPENAI_API_KEY=your-api-key-here
   ```

## Usage

### 1. Data Generation

You can generate synthetic data using either Faker or OpenAI's GPT model:

- Using Faker:
  ```
  python generate_nlq_dataset.py
  ```

- Using OpenAI's GPT model:
  ```
  python generate_nlq_openai.py
  ```

Both scripts will generate a JSON file containing the NLQ pairs.

### 2. Data Preparation

Prepare the generated data for fine-tuning:
This script will create `train_data.txt` and `val_data.txt` files.

### 3. Fine-tuning

Fine-tune the LLaMA model on the prepared dataset:
This script will fine-tune the model and save it in the `./fine_tuned_llama3` directory.

### 4. Inference

Run the inference script with the Gradio interface:

This will launch a web interface where you can enter natural language queries and receive generated SQL queries.

### 5. Model Evaluation

Evaluate the fine-tuned model using various metrics:
`python evaluate_model.py`
This script will calculate BLEU scores, ROUGE scores, and exact match accuracy for a subset of the test data and save the results to `evaluation_results.json`.


## Model Details

This project uses the LLaMA 3 model, specifically the 8B parameter version. The model is fine-tuned using Parameter-Efficient Fine-Tuning (PEFT) with Low-Rank Adaptation (LoRA) to reduce memory requirements and training time.

## Data Generation

The project offers two methods for generating training data:

1. Faker-based generation (`generate_nlq_dataset.py`): This method creates a diverse set of synthetic queries using the Faker library. It generates 10,000 query pairs covering various query types such as top-n, filtering, aggregation, joins, date ranges, grouping, and ordering.

2. OpenAI GPT-based generation (`generate_nlq_openai.py`): This method uses OpenAI's GPT model via the Langchain library to generate more complex and realistic query pairs. It generates 100 pairs by default, but this can be adjusted.

## Fine-tuning Process

The fine-tuning process (`finetune_llama.py`) uses the following techniques:

- 4-bit quantization for memory efficiency
- LoRA for parameter-efficient fine-tuning
- Mixed precision training (fp16)
- Gradient accumulation

The script splits the data into training and validation sets, and uses the Hugging Face Trainer for the fine-tuning process.

## Inference

The inference script (`inference_llama.py`) loads the fine-tuned model and provides a Gradio interface for easy interaction. Users can input natural language queries and receive generated SQL queries in real-time.

## Evaluation

The evaluation script (`evaluate_model.py`) assesses the performance of the fine-tuned model using the following metrics:

- BLEU score: Measures the similarity between the generated SQL query and the reference SQL query.
- ROUGE-L score: Evaluates the longest common subsequence between the generated and reference SQL queries.
- Exact match accuracy: Calculates the percentage of generated SQL queries that exactly match the reference queries.


## Requirements

See `requirements.txt` for a full list of required packages. Key dependencies include:

- transformers
- torch
- peft
- bitsandbytes
- accelerate
- gradio
- faker
- langchain (for OpenAI data generation)

## Notes

- Ensure you have access to the LLaMA model on Hugging Face before running the fine-tuning script.
- The fine-tuning process requires significant computational resources. A GPU with at least 16GB of VRAM is recommended.
- The quality of generated SQL queries depends on the quality and diversity of the training data and the fine-tuning process.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions to this project are welcome! Please follow these steps to contribute:

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

Please ensure your code adheres to the project's coding standards and include tests for new features.

## Contact

For questions, issues, or suggestions, please open an issue on the GitHub repository.
