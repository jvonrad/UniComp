import requests
import json

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "/home/geiger/gwb082/Jonathans_Thesis/compressed-models/quantized/Qwen3-8B-awq"

messages = []

while True:
    user = input("You: ")
    messages.append({"role": "user", "content": user})

    resp = requests.post(
        URL,
        headers={"Content-Type": "application/json"},
        json={"model": MODEL, "messages": messages},
    )

    reply = resp.json()["choices"][0]["message"]["content"]
    print("Model:", reply)
    messages.append({"role": "assistant", "content": reply})
