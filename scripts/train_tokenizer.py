import os
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

def train_tokenizer():
    data_dir = Path("data/cleaned")
    model_dir = Path("models/finance-tokenizer")
    model_dir.mkdir(parents=True, exist_ok=True)

    files = [str(f) for f in data_dir.rglob("*.txt")]
    print(f"Found {len(files)} files. Training custom BPE Tokenizer...")

    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Train it on our corpus
    tokenizer.train(
        files=files,
        vocab_size=32000,
        min_frequency=3,
        special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ]
    )

    # Save the base tokenizer model
    tokenizer.save_model(str(model_dir))

    # Wrap it in HuggingFace PreTrainedTokenizerFast for standard usage
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer._tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>"
    )
    hf_tokenizer.save_pretrained(str(model_dir))
    
    print(f"Custom tokenizer successfully saved to {model_dir}")

if __name__ == "__main__":
    train_tokenizer()
