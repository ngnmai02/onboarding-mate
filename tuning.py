"""
This function is to retrieve and fine-tune the model with new dataset. 
"""
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import json

def main():
    model_name = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Important for GPT-2
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    # Convert JSON lines to plain text
    train_path = 'wikitext_train_sample.json'
    with open(train_path, 'r') as f:
        data = [json.loads(line) for line in f]
    with open('train.txt', 'w') as f:
        for item in data:
            f.write(item['text'] + '\n')

    # Load dataset using Hugging Face Datasets
    dataset = load_dataset('text', data_files={'train': 'train.txt'})
    dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)  # 90% train, 10% validation

    tokenized_dataset = dataset.map(
        lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=128),
        batched=True
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir='./finetuned-onboarding-model',
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        prediction_loss_only=True,
        learning_rate=5e-5
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'], #train
        eval_dataset=tokenized_dataset['test'], #val
        data_collator=data_collator
    )

    trainer.train()
    trainer.save_model('./finetuned-onboarding-model')
    tokenizer.save_pretrained('./finetuned-onboarding-model')
    print('Model fine-tuned and saved to ./finetuned-onboarding-model')

if __name__ == '__main__':
    main()
