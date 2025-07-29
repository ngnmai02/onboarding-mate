"""
This function is used to load the dataset from HuggingFace and save it in a format of choice (e.g. json)
"""
from datasets import load_dataset

def main():
    # load dataset (the smallest one to test)
    dataset = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1')
    # Save a small sample for quick fine-tuning
    train_sample = dataset['train'].select(range(1000))
    train_sample.to_json('wikitext_train_sample.json')
    print('Sample data saved to wikitext_train_sample.json')

if __name__ == '__main__':
    main() 