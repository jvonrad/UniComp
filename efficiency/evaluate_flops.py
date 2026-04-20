from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers.activations as _activations
import argparse
from calflops import calculate_flops

# Compatibility shim for AWQ on newer transformers where PytorchGELUTanh was removed
if not hasattr(_activations, "PytorchGELUTanh"):
    class PytorchGELUTanh(_activations.GELUActivation):
        def __init__(self):
            super().__init__(approximate="tanh")

    _activations.PytorchGELUTanh = PytorchGELUTanh

def main(args):
    # Load tokenizer - for Llama 3, use standard settings
    tokenizer = AutoTokenizer.from_pretrained(
        args.path,
        use_fast=True,
        trust_remote_code=False
    )
    
    # Ensure pad_token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.path,
        device_map="auto",
        trust_remote_code=False
    )
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Calculate FLOPs with correct parameter name
    flops, macs, params = calculate_flops(
        model=model,
        input_shape=(1, args.seqlen),
        transformer_tokenizer=tokenizer,  # Note: transformer_tokenizer, not tokenizer!
        output_precision=2
    )
    
    print(f"\nFLOPs: {flops}")
    print(f"MACs:  {macs}")
    print(f"Params: {params}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help='model checkpoint location')
    parser.add_argument("--seqlen", type=int, default=128, help='sequence length')
    args = parser.parse_args()
    main(args)
