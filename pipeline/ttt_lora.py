import json
import torch
import argparse
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import Dataset
from huggingface_hub import login
import os
from pathlib import Path
import shutil

# Create necessary directories
os.makedirs('input_data', exist_ok=True)
os.makedirs('qa_output', exist_ok=True)

def save_list_to_json(lst, filename):
    """Save results to JSON file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as file:
        json.dump(lst, file, indent=4)

def transform_json(input_json):
    if isinstance(input_json, list):
        return [transform_json(item) for item in input_json]
    new_retrieval_list = []
    for item in input_json["retrieval_list"]:
        new_text = (
            f"[Excerpt from document]\n"
            f"title: {item['metadata']['title']}\n"
            # f"published_at: {item['metadata']['published_at']}\n"
            # f"source: {item['metadata']['source']}\n"
            f"Excerpt:\n"
            f"-----\n"
            f"{item['text']}\n"
            f"-----"
        )
        new_retrieval_list.append({
            "text": new_text,
            "score": item["score"]
        })

    new_gold_list = []
    for item in input_json["gold_list"]:
        new_gold_list.append({
            "title": item["title"],
            "author": item["author"],
            "url": item["url"],
            "source": item["source"],
            "category": item["category"],
            "published_at": item["published_at"],
            "fact": item["fact"]
        })

    new_json = {
        "query": input_json["query"],
        "answer": input_json["answer"],
        "question_type": input_json["question_type"],
        "retrieval_list": new_retrieval_list,
        "gold_list": new_gold_list
    }

    return new_json

def prepare_training_data(doc_data):
    """Prepare training data from retrieved content"""
    training_texts = []
    
    for item in doc_data['retrieval_list']:
        context = item['text']
        training_text = f"""### Instruction:
Use the following context to answer questions accurately.

### Question:
{doc_data['query']}

### Context:
{context}"""
        training_texts.append(training_text)
    
    return Dataset.from_dict({"text": training_texts})

def setup_model_and_tokenizer(model_name, load_in_4bit=True):
    """Initialize model and tokenizer with quantization"""
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True
        )
        torch_dtype = torch.float16
    else:
        quantization_config = None
        torch_dtype = None

    device_map = "auto" if torch.cuda.is_available() else "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def fine_tune_model(model, tokenizer, dataset, output_dir="output"):
    """Fine-tune model using LoRA"""
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"]
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=16,
        learning_rate=3e-4,
        logging_steps=10,
        save_strategy="no"
    )

    peft_model = get_peft_model(model, peft_config)
    
    trainer = SFTTrainer(
        model=peft_model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        max_seq_length=256,
        dataset_text_field="text"
    )
    
    trainer.train()
    return peft_model

def query_model(prompt, model, tokenizer, temperature=0.1, max_new_tokens=512):
    """Generate response from the model"""
    prefix = """You are a precise question-answering assistant. Your task is to answer questions based solely on the provided context. Follow these guidelines:

1. Give direct, concise answers without explanations
2. If the context clearly supports a yes/no answer, respond with just 'Yes' or 'No'
3. For factual questions, provide the specific entity, name, or value
4. If the context doesn't contain enough information to answer confidently, respond with 'Insufficient Information'
5. Base your answer only on the given context, not on external knowledge"""
    
    messages = [
        {"role": "system", "content": prefix},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    model_inputs = tokenizer([text], return_tensors="pt")
    model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs["input_ids"], generated_ids)
    ]

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def process_single_query(query_data, base_model, tokenizer, output_dir):
    """Process a single query with fine-tuning and inference"""
    # Prepare training data
    dataset = prepare_training_data(query_data)
    
    # Fine-tune model
    fine_tuned_model = fine_tune_model(base_model, tokenizer, dataset, output_dir)
    
    # Prepare inference prompt
    prompt = f"Answer the following question: {query_data['query']}"
    
    # Get model response
    response = query_model(prompt, fine_tuned_model, tokenizer)
    
    # Clear CUDA cache and delete model
    del fine_tuned_model
    torch.cuda.empty_cache()
    
    # Clean up output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    return {
        'query': query_data['query'],
        'prompt': prompt,
        'model_answer': response,
        'gold_answer': query_data['answer'],
        'question_type': query_data['question_type']
    }

def main():
    parser = argparse.ArgumentParser(description='Fine-tune and inference with Llama model')
    parser.add_argument('--hf_token', type=str, required=True, help='Hugging Face login token')
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf", help='Model name')
    parser.add_argument('--input_files', nargs='+', help='Specific input files to process')
    
    args = parser.parse_args()

    # Clone repository
    os.system("git clone https://github.com/vlgiitr/ttt-icl-math.git")
    
    # Set up CUDA
    torch.set_default_dtype(torch.float16)
    
    # Login to Hugging Face
    login(token=args.hf_token)

    print("Loading base model and tokenizer...")
    base_model, tokenizer = setup_model_and_tokenizer(args.model_name)
    print("Model and tokenizer loaded successfully!")

    # Process specified input files
    input_dir = Path('input_data')
    if args.input_files:
        json_files = [input_dir / filename for filename in args.input_files]
    else:
        json_files = list(input_dir.glob('*.json'))
    
    if not json_files:
        print("No JSON files found to process")
        return

    print(f"Processing {len(json_files)} files")
    
    for input_file in json_files:
        print(f"\nProcessing file: {input_file.name}")
        
        try:
            with open(input_file, 'r') as file:
                data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error processing {input_file}: {str(e)}")
            continue

        doc_data = transform_json(data)
        save_list = []
        
        # Process each query in the file
        for d in tqdm(doc_data, desc=f"Processing queries in {input_file.name}"):
            result = process_single_query(d, base_model, tokenizer, "output")
            save_list.append(result)
        
        # Save results
        output_filename = f'qa_output/llama_{input_file.stem}_output.json'
        save_list_to_json(save_list, output_filename)
        print(f"Results saved to {output_filename}")

    print("\nAll files processed successfully!")

if __name__ == "__main__":
    main()