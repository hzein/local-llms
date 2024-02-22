import numpy as np
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding

dataset = load_dataset("amazon_us_reviews", "Electronics_v1_00", split="train")

wanted_features = ["review_body", "review_headline", "product_title", "star_rating", "verified_purchase"]

dataset = dataset.remove_columns(x for x in dataset.features if x not in wanted_features)

dataset = dataset.filter(lambda x: x['verified_purchase'] and len(x['review_body']) > 100)
dataset = dataset.shuffle(seed=42).select(range(100000))

dataset = dataset.class_encode_column("star_rating")
dataset = dataset.train_test_split(test_size=0.1, seed=42, stratify_by_column="star_rating")

train_dataset = dataset['train']
test_dataset = dataset['test']

model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)

def preprocess_data(examples):
    examples['prompt'] = [f"review: {product_title}, {star_rating} Stars!" for product_title, star_rating in zip(examples['review_body'], examples['star_rating'])]
    examples['response'] = [f"{review_headline} {review_body}" for review_headline, review_body in zip(examples['review_headline'], examples['review_body'])]

    inputs = tokenizer(examples['prompt'], padding="max_length", truncation=True, max_length=128)
    target = tokenizer(examples['response'], padding="max_length", truncation=True, max_length=128)

    inputs.update({'labels': target['input_ids']})

    return inputs

train_dataset = train_dataset.map(preprocess_data, batched=True)
test_dataset = test_dataset.map(preprocess_data, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = T5ForConditionalGeneration.from_pretrained(model_name).to("cuda")

training_args = TrainingArguments(
    output_dir="./t5-fine-tuned-reviews",
    num_train_epochs=3,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

print("Training...")
trainer.train()
trainer.save_model()