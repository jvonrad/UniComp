# quantize_model.py
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

def quantize(model_path: str, output_dir: str, bits: int = 4, dataset="c4"):
    # load tokenizer & set up quantization config
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    gptq_config = GPTQConfig(bits=bits, dataset=dataset, tokenizer=tokenizer)
    
    max_memory = {
        0: "78GiB",   # GPU 0
        1: "78GiB",   # GPU 1
        2: "78GiB",   # GPU 2
        3: "78GiB",   # GPU 3
        "cpu": "200GiB"  # oder so viel, wie auf deinem Host zur Verfügung steht
    }
    
    # load and quantize the model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=gptq_config,
        max_memory=max_memory,
        trust_remote_code=True
    )
    # move to CPU and save
    model.to("cpu")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Quantized {model_path} → {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="llama-2-7b-hf", help="Path or HF repo of the model to quantize")
    parser.add_argument("--output_dir", default="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/quantized/llama_2_7b_gptq4bit", help="Directory to save the quantized model")
    parser.add_argument("--bits", type=int, default=4, help="Number of quantization bits")
    args = parser.parse_args()
    quantize(args.model_path, args.output_dir, bits=args.bits)

