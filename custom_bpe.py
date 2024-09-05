from collections import Counter
import re
from datasets import load_dataset
from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast
def create_bpe_tokenizer(model,text_corpus):
    # Initialize a tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained(model)
    
    # Assuming your text corpus is in a text file
    dataset = load_dataset('text', data_files={'train': text_corpus})
    
    with open(text_corpus, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Simple tokenization (splitting by whitespace and punctuation)
    tokens = re.findall(r'\b\w+\b', text.lower())
    
    token_counts = Counter(tokens)
    unique_tokens = len(token_counts)
    print(f"Number of unique tokens: {unique_tokens}")
    
    # Train the tokenizer on your dataset
    tokenizer.train_new_from_iterator(dataset['train']['text'], vocab_size=unique_tokens)
    return tokenizer
if __name__ == "__main__":
    tokenizer = create_bpe_tokenizer("Inoob/NullGPT2", "cleared.txt")
    tokenizer.save_pretrained('BibleArchitecture-Tokenizer')

