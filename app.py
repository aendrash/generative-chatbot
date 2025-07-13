from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Load fine-tuned model and tokenizer
model_path = "./gpt2-faq"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message', '')

    # Prepare input prompt
    prompt = f"Q: {user_input} A:"
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Generate response
    outputs = model.generate(
        inputs,
        max_length=150,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        early_stopping=True
    )

    # Decode and clean
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the answer after 'A:'
    if "A:" in full_output:
        answer = full_output.split("A:")[1].split("Q:")[0].strip()
    else:
        answer = full_output.strip()

    return jsonify({"response": answer})


@app.route('/', methods=['GET'])
def home():
    return "Flask server is running"


if __name__ == '__main__':
    app.run(debug=True)
