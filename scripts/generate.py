import torch
from torch.nn import functional as F
from pathlib import Path
from transformers import PreTrainedTokenizerFast
from train_model import GPTLanguageModel, block_size

def generate_text(model, idx, max_new_tokens):
    """
    Autoregressively predicts the next token by rolling a mathematical probability 
    distribution over the neural network's outputs.
    """
    for _ in range(max_new_tokens):
        # Crop context to the maximum block size the model was trained on
        idx_cond = idx[:, -block_size:]
        
        # Feed into the neural network to get the predictions
        logits, _ = model(idx_cond)
        
        # Focus strictly on the very last predicted time step
        logits = logits[:, -1, :] # Becomes (Batch, Vocab_Size)
        
        # Convert raw network logits into mathematical probabilities
        probs = F.softmax(logits, dim=-1) 
        
        # Sample the next token from the probability distribution
        idx_next = torch.multinomial(probs, num_samples=1) 
        
        # Append to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) 
    return idx

def main():
    print("Loading PyTorch & Initializing Hardware...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Loading Custom Financial Tokenizer...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained("models/finance-tokenizer")
    
    print("Loading Base Model Architecture...")
    model = GPTLanguageModel()
    
    print("Injecting Custom Neural Weights...")
    model.load_state_dict(torch.load("models/finance-llm-custom/model.pt", map_location=device, weights_only=True))
    model.to(device)
    model.eval() # Set model into strict evaluation mode (disables dropout layers)
    
    print("\n================== FINANCE-LM INFERENCE ==================")
    print("Type a prompt to seed the neural network (or 'quit' to exit)\n")
    
    while True:
        prompt = input("Prompt: ")
        if prompt.lower() in ['quit', 'exit']:
            break
            
        # Encode the string into integer tokens using our BPE tokenizer
        context = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        
        print("\nGenerating...")
        with torch.no_grad(): # Disable gradient tracking to save massive amounts of VRAM
            generated_tokens = generate_text(model, context, max_new_tokens=150)
            
        # Decode the generated integer tokens back into human-readable strings
        output_text = tokenizer.decode(generated_tokens[0].tolist())
        print(f"\n[Output]:\n{output_text}\n")
        print("="*60 + "\n")

if __name__ == "__main__":
    main()
