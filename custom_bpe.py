from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from collections import Counter

# Initialize a tokenizer
tokenizer = Tokenizer(models.BPE())

# Customize pre-tokenization and post-processing
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel()
def create_bpe_tokenizer(text_file):
    # Load your text corpus
    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = text.split()

    # Count unique tokens
    vocab = Counter(tokens)
    vocab_size = len(vocab)

    print(f"Vocabulary size: {vocab_size}")
    # Train the tokenizer
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    tokenizer.eos_token = "</s>"
    tokenizer.bos_token = "<s>"
    tokenizer.unk_token = "<unk>"
    tokenizer.pad_token = "<pad>"
    tokenizer.mask_token = "<mask>"
    tokenizer.train_from_iterator([text], trainer)
    return tokenizer
if __name__ == "__main__":
    tokenizer=create_bpe_tokenizer("cleared.txt")
    # Save the tokenizer
    tokenizer.save("custom_bpe_tokenizer.json")
