"""
Inference script for the fine-tuned Edmond Dantès model

Usage:
    python inference_dantes.py --model_path outputs/dantes_lora
"""

import os
# Set Triton PTXAS path before importing CUDA-dependent libraries
os.environ['TRITON_PTXAS_PATH'] = '/usr/local/cuda/bin/ptxas'
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import TextStreamer

def load_model(model_path, max_seq_length=2048):
    """Load the fine-tuned model"""
    print(f"📥 Loading model from {model_path}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    # Setup chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
        map_eos_token=True,
    )

    # Fix pad token to avoid attention mask warning
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Enable fast inference
    FastLanguageModel.for_inference(model)

    print("✅ Model loaded successfully!\n")
    return model, tokenizer

def generate_response(model, tokenizer, user_message, max_new_tokens=256, stream=True):
    """Generate a response from the model"""

    messages = [
        {"from": "human", "value": user_message}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    # Create attention mask
    attention_mask = (inputs != tokenizer.pad_token_id).long()

    if stream:
        # Stream the output token by token, skipping special tokens
        text_streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        _ = model.generate(
            input_ids=inputs,
            attention_mask=attention_mask,
            streamer=text_streamer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    else:
        # Generate without streaming
        outputs = model.generate(
            input_ids=inputs,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the assistant's response
        response = response.split("<|im_start|>assistant\n")[-1].strip()
        print(response)

def interactive_mode(model, tokenizer):
    """Run an interactive conversation"""
    print("=" * 60)
    print("🎭 Conversation avec Edmond Dantès")
    print("=" * 60)
    print("Tapez 'exit' ou 'quit' pour quitter\n")

    while True:
        try:
            user_input = input("\n👤 Vous: ").strip()

            if user_input.lower() in ["exit", "quit", "q"]:
                print("\n👋 Au revoir!")
                break

            if not user_input:
                continue

            print("\n🎭 Edmond Dantès: ", end="", flush=True)
            generate_response(model, tokenizer, user_input, stream=True)

        except KeyboardInterrupt:
            print("\n\n👋 Au revoir!")
            break
        except Exception as e:
            print(f"\n❌ Erreur: {e}")

def test_examples(model, tokenizer):
    """Test with example questions"""
    examples = [
        "Que pensez-vous de la justice?",
        "Comment vous sentez-vous après votre emprisonnement?",
        "Parlez-moi de votre vengeance.",
        "Quelle est votre philosophie de la vie?",
    ]

    print("=" * 60)
    print("🧪 Testing with example questions")
    print("=" * 60)

    for i, question in enumerate(examples, 1):
        print(f"\n[{i}/{len(examples)}] Question: {question}")
        print("\n🎭 Edmond Dantès: ", end="", flush=True)
        generate_response(model, tokenizer, question, stream=True)
        print()

def main(args):
    # Load the model
    model, tokenizer = load_model(args.model_path, args.max_seq_length)

    if args.test:
        # Run test examples
        test_examples(model, tokenizer)
    elif args.question:
        # Single question mode
        print(f"👤 Question: {args.question}\n")
        print("🎭 Edmond Dantès: ", end="", flush=True)
        generate_response(model, tokenizer, args.question, max_new_tokens=args.max_tokens, stream=True)
    else:
        # Interactive mode
        interactive_mode(model, tokenizer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with Edmond Dantès model")

    parser.add_argument("--model_path", type=str, default="outputs/dantes_lora",
                        help="Path to the fine-tuned model")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="Maximum tokens to generate")
    parser.add_argument("--test", action="store_true",
                        help="Run test examples")
    parser.add_argument("--question", type=str,
                        help="Ask a single question")

    args = parser.parse_args()

    main(args)
