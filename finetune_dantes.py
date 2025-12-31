"""
Fine-tuning Mistral 7B to act as Edmond Dantès using Unsloth
Based on Unsloth's conversational fine-tuning template

Usage:
    python finetune_dantes.py --dataset data/dataset/dantes_conversations.jsonl --output_dir outputs/dantes_model
"""

import argparse
import time
import sys
from datetime import datetime
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import torch

class TeeLogger:
    """Logger that writes to both terminal and file"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

def main(args):
    # Setup logging to both terminal and file
    log_filename = f"{args.output_dir}_training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    logger = TeeLogger(log_filename)
    sys.stdout = logger
    sys.stderr = logger

    # Start timing
    script_start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Model configuration
    max_seq_length = args.max_seq_length  # Configurable sequence length
    dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True  # Use 4bit quantization to reduce memory usage

    print("=" * 60)
    print("🎭 Fine-tuning Mistral 7B as Edmond Dantès")
    print("=" * 60)
    print(f"⏰ Started at: {start_datetime}")
    print(f"📝 Log file: {log_filename}")

    # Load the base model
    print("\n📥 Loading base model: mistral-7b-instruct-v0.3...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # Add LoRA adapters
    print("\n🔧 Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,  # Optimized for 0
        bias="none",  # Optimized for "none"
        use_gradient_checkpointing="unsloth",  # Use Unsloth's optimized checkpointing
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    model_load_time = time.time() - script_start_time
    print(f"✅ Model setup complete in {model_load_time:.2f} seconds ({model_load_time/60:.2f} minutes)")

    # Setup chat template
    print("\n📝 Setting up ChatML template...")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",  # Using ChatML format
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
        map_eos_token=True,  # Maps <|im_end|> to </s>
    )

    def formatting_prompts_func(examples):
        """Convert conversations to chat template format"""
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            for convo in convos
        ]
        return {"text": texts}

    # Load dataset
    print(f"\n📚 Loading dataset from {args.dataset}...")
    dataset = load_dataset("json", data_files=args.dataset, split="train")
    print(f"   Dataset size: {len(dataset)} conversations")

    # Apply formatting
    dataset = dataset.map(formatting_prompts_func, batched=True)

    # Show a sample
    if len(dataset) > 0:
        print("\n📄 Sample formatted conversation:")
        print("-" * 60)
        print(dataset[0]["text"])
        print("-" * 60)

    dataset_load_time = time.time() - script_start_time - model_load_time
    print(f"\n✅ Dataset loaded and formatted in {dataset_load_time:.2f} seconds ({dataset_load_time/60:.2f} minutes)")

    # Training configuration
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    print("\n🎯 Training configuration:")
    print(f"   Batch size per device: {args.batch_size}")
    print(f"   Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"   Effective batch size: {effective_batch_size}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Num epochs: {args.num_epochs if not args.max_steps else 'N/A'}")
    print(f"   Max steps: {args.max_steps if args.max_steps else 'Full epoch'}")
    print(f"   Max sequence length: {max_seq_length}")
    print(f"   Packing: {'Enabled' if args.packing else 'Disabled'}")
    print(f"   Output directory: {args.output_dir}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=args.packing,  # Configurable packing
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps if args.max_steps else -1,
            num_train_epochs=args.num_epochs if not args.max_steps else 1,
            learning_rate=args.learning_rate,
            logging_steps=args.logging_steps,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=args.output_dir,
            report_to="none",  # Set to "wandb" or "tensorboard" for tracking
            dataloader_num_workers=2,  # Parallel data loading (conservative for 4bit)
            dataloader_pin_memory=False,  # Not needed for unified memory
            dataloader_prefetch_factor=2,  # Prefetch batches to avoid GPU starvation
        ),
    )

    # Show memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"\n💾 GPU: {gpu_stats.name}")
    print(f"   Max memory: {max_memory} GB")
    print(f"   Reserved: {start_gpu_memory} GB")

    # Train
    print("\n🚀 Starting training...")
    print("=" * 60)
    training_start_time = time.time()
    trainer_stats = trainer.train()
    training_end_time = time.time()
    training_time = training_end_time - training_start_time

    # Show final stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)

    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print("=" * 60)
    print(f"⏱️  Training time (measured): {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"⏱️  Training time (reported): {round(trainer_stats.metrics['train_runtime'], 2)} seconds ({round(trainer_stats.metrics['train_runtime']/60, 2)} minutes)")
    print(f"💾 Peak memory: {used_memory} GB ({used_percentage}% of total)")
    print(f"📊 LoRA memory: {used_memory_for_lora} GB")

    # Save the model
    print(f"\n💾 Saving model to {args.output_dir}...")
    save_start_time = time.time()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if args.save_merged:
        print(f"\n💾 Saving merged 16-bit model to {args.output_dir}_merged...")
        model.save_pretrained_merged(
            f"{args.output_dir}_merged",
            tokenizer,
            save_method="merged_16bit"
        )

    if args.save_gguf:
        print(f"\n💾 Saving GGUF model to {args.output_dir}_gguf...")
        model.save_pretrained_gguf(
            f"{args.output_dir}_gguf",
            tokenizer,
            quantization_method=args.gguf_quantization
        )

    save_time = time.time() - save_start_time
    print(f"✅ Model saved in {save_time:.2f} seconds ({save_time/60:.2f} minutes)")

    # Total script time
    total_time = time.time() - script_start_time
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\n" + "=" * 60)
    print("🎉 All done! Model saved successfully.")
    print("=" * 60)
    print(f"⏰ Started at:  {start_datetime}")
    print(f"⏰ Finished at: {end_datetime}")
    print(f"⏱️  Total time:  {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print("\n📊 Time breakdown:")
    print(f"   Model setup:     {model_load_time:>8.2f}s ({model_load_time/60:>6.2f}m) - {model_load_time/total_time*100:>5.1f}%")
    print(f"   Dataset loading: {dataset_load_time:>8.2f}s ({dataset_load_time/60:>6.2f}m) - {dataset_load_time/total_time*100:>5.1f}%")
    print(f"   Training:        {training_time:>8.2f}s ({training_time/60:>6.2f}m) - {training_time/total_time*100:>5.1f}%")
    print(f"   Saving:          {save_time:>8.2f}s ({save_time/60:>6.2f}m) - {save_time/total_time*100:>5.1f}%")
    print(f"\n📝 To use the model for inference, load it with:")
    print(f"   FastLanguageModel.from_pretrained('{args.output_dir}')")

    # Close the logger
    logger.close()
    sys.stdout = logger.terminal
    sys.stderr = sys.__stderr__

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Mistral 7B as Edmond Dantès")

    # Data arguments
    parser.add_argument("--dataset", type=str, default="data/dataset/dantes_conversations.jsonl",
                        help="Path to the JSONL dataset")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1.875e-4,
                        help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--warmup_steps", type=int, default=5,
                        help="Warmup steps")
    parser.add_argument("--packing", action="store_true", default=True,
                        help="Enable sequence packing (default: True)")
    parser.add_argument("--no-packing", dest="packing", action="store_false",
                        help="Disable sequence packing")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Maximum training steps (overrides num_epochs)")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs (used if max_steps is None)")
    parser.add_argument("--logging_steps", type=int, default=1,
                        help="Logging frequency")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="outputs/dantes_lora",
                        help="Output directory for LoRA adapters")
    parser.add_argument("--save_merged", action="store_true",
                        help="Save merged 16-bit model")
    parser.add_argument("--save_gguf", action="store_true",
                        help="Save GGUF quantized model")
    parser.add_argument("--gguf_quantization", type=str, default="q4_k_m",
                        choices=["q8_0", "q4_k_m", "q5_k_m", "f16"],
                        help="GGUF quantization method")

    args = parser.parse_args()

    main(args)
