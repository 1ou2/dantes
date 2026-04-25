# Dantès - Fine-tuning Mistral 7B as Edmond Dantès

Fine-tuning project to create a model that acts as Edmond Dantès from "The Count of Monte Cristo". The model is trained on character-specific data extracted from the original French text.

**Target model:** mistral-7b-instruct

## Quick Start

```bash
# Install dependencies
uv venv
source .venv/bin/activate
./setup.sh

# Run the complete pipeline (see Data Pipeline section below)
python src/gutenberg.py
python src/citation.py
python src/instructions.py
python src/prepare_dataset.py instructions-result.txt

# Fine-tune the model
python finetune_dantes.py

# Use the model
python inference_dantes.py --model_path outputs/dantes_lora
```

## Running Inference (without fine-tuning locally)

If you want to run inference directly without going through the full data pipeline and fine-tuning process, you can download the pre-trained LoRA adapter from Hugging Face.

### 1. Install dependencies

```bash
uv venv
source .venv/bin/activate
./setup.sh
```

### 2. Authenticate with Hugging Face

```bash
pip install huggingface_hub[cli]
hf auth login
```

### 3. Download the LoRA adapter

```bash
hf download 1ou2/comte-monte-cristo-mistral-7b --local-dir outputs/dantes_lora
```

This downloads the fine-tuned LoRA adapter files (`adapter_model.safetensors`, `adapter_config.json`, tokenizer files, etc.) into `outputs/dantes_lora/`.

> **Note:** The base model `unsloth/mistral-7b-instruct-v0.3-bnb-4bit` is downloaded automatically by unsloth on first run. No manual download is required for it.

### 4. Run the base model (before fine-tuning)

This runs the original Mistral 7B Instruct model without any fine-tuning, useful for comparison:

```bash
# Interactive mode
python inference_base.py

# Test with predefined questions
python inference_base.py --test

# Ask a single question
python inference_base.py --question "Que pensez-vous de la justice?"
```

### 5. Run the fine-tuned Dantès model

Using the downloaded LoRA adapter:

```bash
# Interactive mode (uses outputs/dantes_lora by default)
python inference_dantes.py

# Or specify the path explicitly
python inference_dantes.py --model_path outputs/dantes_lora

# You can also point directly to the Hugging Face repo (downloads on the fly)
python inference_dantes.py --model_path 1ou2/comte-monte-cristo-mistral-7b

# Test with predefined questions
python inference_dantes.py --test

# Ask a single question
python inference_dantes.py --question "Que pensez-vous de la justice?"
```

### Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support (the model runs in 4-bit quantization)
- **VRAM**: ~8 GB minimum
- **CUDA**: Compatible CUDA toolkit installed

## Prerequisites

### Local LLM Server
All data generation scripts require a local LLM server running:
- **URL:** `http://0.0.0.0:12001/v1/chat/completions`
- **Model:** `gpt-oss-20b-default`
- **Server:** llama.cpp or compatible OpenAI-style API

Start the server before running data generation scripts.

### Hugging Face CLI (optional)
```bash
# Install
curl -LsSf https://hf.co/cli/install.sh | bash

# Login and download models
hf auth login
hf download mistralai/Ministral-3-3B-Instruct-2512
```

## Data Pipeline

The project follows a sequential pipeline to generate fine-tuning data:

### 1. Download and Process Source Text
**Script:** `src/gutenberg.py`

Downloads and preprocesses the original French text of "Le Comte de Monte-Cristo" from Project Gutenberg.

**Process:**
- Downloads 4 Gutenberg files (IDs: 17989, 17990, 17991, 17992)
- Removes UTF-8 BOM, headers, and footers
- Reformats text by removing hard line breaks around column 80
- Preserves paragraph structure and dialogues
- Special handling for dialogue markers (—, --, –, -, «)

**Output:** `data/gutenberg/*.txt` (cleaned, reformatted text files)

```bash
python src/gutenberg.py
```

### 2. Extract Character Citations
**Script:** `src/citation.py`

Extracts Edmond Dantès' thoughts, speech, and writings from the processed text.

**Process:**
- Splits text into 100-line chunks with 10-line overlap
- Sends each chunk to local LLM with extraction prompt
- LLM identifies and extracts Dantès-specific content
- Returns structured data: `{"contexte": "...", "citation": "..."}`

**Output:** `data/citations/dantes.jsonl` (JSONL file with character citations)

```bash
python src/citation.py
```

**Note:** Adjust `max_chunks` and `start_chunk` in the script to process specific ranges.

### 3. Generate Instruction Dataset
**Script:** `src/instructions.py`

Converts citations into question-answer pairs for instruction tuning.

**Process:**
- For each citation, generates 3 Q&A pairs
- Maintains Dantès' 19th-century formal French style
- Creates instruction-tuning format: `{"instruction": "...", "response": "..."}`

**Output:** `instructions-result.txt` (raw LLM output with multiple JSON objects)

```bash
python src/instructions.py
```

**Modes:**
- `sync_main()`: Sequential processing with progress bar
- `main()`: Async processing (max 3 concurrent requests)

### 4. Prepare Training Dataset
**Script:** `src/prepare_dataset.py`

Transforms raw instruction data into proper training format.

**Process:**
- Parses multiple JSON objects per line from LLM output
- Converts to ShareGPT-style conversation format
- Creates two versions: basic and with system prompt

**Output:**
- `data/dataset/dantes_conversations.jsonl` (basic format)
- `data/dataset/dantes_conversations_system.jsonl` (with system prompt)

```bash
# For testing
python src/prepare_dataset.py instructions-sample.txt

# For full dataset
python src/prepare_dataset.py instructions-result.txt
```

**Format:**
```json
{
  "conversations": [
    {"from": "human", "value": "question"},
    {"from": "gpt", "value": "response"}
  ]
}
```

### 5. Fine-tune the Model
**Script:** `finetune_dantes.py`

Fine-tunes Mistral 7B using Unsloth for efficient training.

**Basic usage:**
```bash
python finetune_dantes.py
```

**Full options:**
```bash
python finetune_dantes.py \
  --dataset data/dataset/dantes_conversations.jsonl \
  --output_dir outputs/dantes_lora \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-4 \
  --num_epochs 3 \
  --save_merged \
  --save_gguf \
  --gguf_quantization q4_k_m
```

**Key parameters:**
- `--dataset`: Training dataset path
- `--batch_size`: Per-device batch size (default: 2)
- `--num_epochs`: Training epochs (default: 3)
- `--output_dir`: LoRA adapter output directory
- `--save_merged`: Save merged 16-bit model for VLLM
- `--save_gguf`: Save quantized GGUF model
- `--gguf_quantization`: GGUF format (q8_0, q4_k_m, q5_k_m, f16)

**Output:** LoRA adapters in `outputs/dantes_lora/`

### 6. Run Inference
**Script:** `inference_dantes.py`

Use the fine-tuned model for interactive conversations.

**Interactive mode:**
```bash
python inference_dantes.py --model_path outputs/dantes_lora
```

**Single question:**
```bash
python inference_dantes.py --model_path outputs/dantes_lora --question "Que pensez-vous de la justice?"
```

**Test examples:**
```bash
python inference_dantes.py --model_path outputs/dantes_lora --test
```

**Parameters:**
- `--model_path`: Path to fine-tuned model (default: outputs/dantes_lora)
- `--max_tokens`: Maximum tokens to generate (default: 256)
- `--test`: Run predefined test questions
- `--question`: Ask single question and exit

## Troubleshooting

### CUDA Architecture Error
If you encounter this error:
```
Error Internal Triton PTX codegen error
`ptxas` stderr:
ptxas fatal   : Value 'sm_121a' is not defined for option 'gpu-name'
```

**Solution:**
```bash
export TORCH_CUDA_ARCH_LIST=12.1a
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
python finetune_dantes.py
```

## Project Structure

```
data/
├── raw/              # Downloaded Gutenberg files
├── preprocessed/     # Header/footer removed
├── gutenberg/        # Final reformatted text
├── citations/        # Extracted citations (JSONL)
│   └── dantes.jsonl
└── dataset/          # Fine-tuning datasets
    ├── dantes_conversations.jsonl
    └── dantes_conversations_system.jsonl

outputs/
└── dantes_lora/      # Trained LoRA adapters
    ├── adapter_config.json
    ├── adapter_model.safetensors
    └── ...

src/
├── gutenberg.py      # Download and process source text
├── citation.py       # Extract character citations
├── charactercard.py  # Generate character profile
├── instructions.py   # Generate Q&A pairs
├── prepare_dataset.py # Format for training
└── llm.py           # LLM wrapper classes
```

## Additional Tools

### Character Card Generation
**Script:** `src/charactercard.py`

Generates a character profile analyzing Dantès' tone, style, and personality based on extracted citations.

**Output:**
- `card-result.txt` (intermediate analyses)
- `character-card-summary.txt` (final character profile)

```bash
python src/charactercard.py
```
