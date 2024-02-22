import torch
import numpy as np
import evaluate

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Cuda version: {torch.version.cuda}")

metric = evaluate.load("accuracy")

def preprocess(example):
    example["prompt"] = f"{example['instruction']} {example['input']} {example['output']}"

    return example

def tokenize_dataset(dataset):
    tokenize_dataset = dataset.map(lambda example:tokenizer(example["prompt"], truncation=True, max_length=128), batched=True, remove_columns=["prompt"])
    
    return tokenize_dataset

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# dataset = load_dataset("hakurei/open-instruct-v1", split="train")

# dataset = dataset.map(preprocess, remove_columns=["instruction", "input", "output"])

# dataset = dataset.shuffle(seed=42).select(range(100000)).train_test_split(test_size=0.1)
# dataset = dataset.select(range(100000)).train_test_split(test_size=0.1)
# dataset = dataset.train_test_split(test_size=0.1)
# train_dataset = dataset['train']
# test_dataset = dataset['test']

model_name = "microsoft/DialoGPT-medium"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# train_dataset = tokenize_dataset(train_dataset)
# test_dataset = tokenize_dataset(test_dataset)

model = AutoModelForCausalLM.from_pretrained("./dialogpt2-instruct").to("cuda")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# training_args = TrainingArguments(
#     output_dir="./dialogpt2-instruct",
#     num_train_epochs=1,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=16,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
# )

# Train Model 

# print("Training...")
# trainer.train(resume_from_checkpoint="./dialogpt2-instruct/checkpoint-53500")
# trainer.save_model()
# print("Model trained and saved!")

# RUN Model

model = AutoModelForCausalLM.from_pretrained("./dialogpt2-instruct/checkpoint-53500").to("cuda")

def generate_text(prompt):
    template = """
    Hassan Zein is a fellow from lebanon, he has been in living in the US for 15 years. 
    He is married to Lisa and has two sons. His two sons names are Liam and Leonidas.
     
    {prompt}
    """
    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs, max_length=64, pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text[:generated_text.rfind('.')+1]

while True:
    prompt = input("Prompt: ")
    if prompt == "exit":
        break
    print("Response:", generate_text(prompt))

# # print(generate_text("What is the best way to cook chicken breast?"))
# # print(generate_text("who won last year champions league?"))

