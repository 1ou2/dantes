#!/bin/bash

# Upload Dantès LoRA model to Hugging Face
# Usage: ./upload_to_hf.sh

set -e  # Exit on error

# Configuration
HF_USERNAME="1ou2"
REPO_NAME="comte-monte-cristo-mistral-7b"
MODEL_PATH="outputs/dantes_lora"
REPO_ID="${HF_USERNAME}/${REPO_NAME}"

echo "============================================================"
echo "📤 Uploading Edmond Dantès model to Hugging Face"
echo "============================================================"
echo "Repository: ${REPO_ID}"
echo "Source path: ${MODEL_PATH}"
echo ""

# Authentication note
echo "🔐 Using existing Hugging Face authentication..."
echo "   (If upload fails, run: hf auth login)"
echo ""

# Create the repository
echo "📦 Creating repository: ${REPO_ID}..."
hf repo create ${REPO_NAME} --type model --exist-ok || {
    echo "⚠️  Repository might already exist, continuing..."
}
echo ""

# Create a temporary directory for upload
TEMP_DIR=$(mktemp -d)
echo "📁 Preparing files in temporary directory: ${TEMP_DIR}"

# Copy essential files
echo "📋 Copying essential model files..."
cp "${MODEL_PATH}/adapter_model.safetensors" "${TEMP_DIR}/"
cp "${MODEL_PATH}/adapter_config.json" "${TEMP_DIR}/"
cp "${MODEL_PATH}/tokenizer.json" "${TEMP_DIR}/"
cp "${MODEL_PATH}/tokenizer_config.json" "${TEMP_DIR}/"
cp "${MODEL_PATH}/special_tokens_map.json" "${TEMP_DIR}/"
cp "${MODEL_PATH}/chat_template.jinja" "${TEMP_DIR}/"

# Copy HF_README.md as README.md
echo "📝 Copying documentation..."
cp "${MODEL_PATH}/HF_README.md" "${TEMP_DIR}/README.md"

echo ""
echo "✅ Files prepared:"
ls -lh "${TEMP_DIR}"
echo ""

# Calculate total size
TOTAL_SIZE=$(du -sh "${TEMP_DIR}" | awk '{print $1}')
echo "📊 Total upload size: ${TOTAL_SIZE}"
echo ""

# Upload files to Hugging Face
echo "🚀 Uploading files to Hugging Face..."
echo "   This may take a few minutes depending on your connection..."
echo ""

hf upload ${REPO_ID} "${TEMP_DIR}" . --repo-type model

echo ""
echo "============================================================"
echo "✅ Upload complete!"
echo "============================================================"
echo "🔗 Your model is now available at:"
echo "   https://huggingface.co/${REPO_ID}"
echo ""
echo "📖 To use your model, run:"
echo ""
echo "   from unsloth import FastLanguageModel"
echo "   model, tokenizer = FastLanguageModel.from_pretrained("
echo "       model_name=\"${REPO_ID}\","
echo "       max_seq_length=2048,"
echo "       dtype=None,"
echo "       load_in_4bit=True,"
echo "   )"
echo ""

# Clean up
rm -rf "${TEMP_DIR}"
echo "🧹 Cleaned up temporary files"
echo ""
echo "🎉 All done!"
