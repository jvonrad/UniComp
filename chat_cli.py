#!/usr/bin/env python3
import sys
import argparse
from vllm import LLM

def parse_args():
    p = argparse.ArgumentParser(description="vLLM interactive CLI chat")
    p.add_argument("--model", type=str, required=True,
                   help="Pfad oder HuggingFace-Repo des Modells")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=512)
    return p.parse_args()

def main():
    args = parse_args()

    # Engine starten
    llm = LLM(
        model=args.model,
        dtype="bfloat16",       # oder float16/float32
        device="cuda",          # oder "cpu"
    )
    # Default-SamplingParams holen und anpassen
    sampling_params = llm.get_default_sampling_params()
    sampling_params.temperature = args.temperature
    sampling_params.max_tokens = args.max_tokens

    # Chat-Historie initialisieren
    history = [
        {"role": "system", "content": "Du bist ein hilfreicher Assistent."}
    ]

    print("=== Starte CLI‐Chat mit vLLM ===")
    print("Tippe deine Nachricht, 'exit' zum Beenden.\n")

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ("exit", "quit"):
            print("Bye!")
            break

        # neue User‐Nachricht anhängen
        history.append({"role": "user", "content": user_input})

        # Chat ausführen
        outputs = llm.chat([history], sampling_params, use_tqdm=False)
        resp = outputs[0].outputs[0].text.strip()

        # Antwort anzeigen und zur Historie hinzufügen
        print(f"Assistant: {resp}\n")
        history.append({"role": "assistant", "content": resp})

if __name__ == "__main__":
    main()
