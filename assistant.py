"""
This function is to set up for fine-tuned assistant bot in command line for usage
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
    model_dir = './finetuned-onboarding-model'
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    print('Onboarding AI Assistant. Type your question (or "exit" to quit):')
    while True:
        question = input('> ')
        if question.lower() == 'exit':
            break
        inputs = tokenizer(question, return_tensors='pt', padding=True, truncation=True)
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=100,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print('Assistant:', answer)

if __name__ == '__main__':
    main() 