#!/bin/bash

# Upload Dantès Conversation Dataset to Hugging Face
# Usage: ./upload_dataset_to_hf.sh

set -e  # Exit on error

# Configuration
HF_USERNAME="1ou2"
REPO_NAME="comte-monte-cristo-conversations"
DATASET_PATH="data/dataset"
REPO_ID="${HF_USERNAME}/${REPO_NAME}"

echo "============================================================"
echo "📤 Uploading Edmond Dantès Conversation Dataset to Hugging Face"
echo "============================================================"
echo "Repository: ${REPO_ID}"
echo "Source path: ${DATASET_PATH}"
echo ""

# Authentication note
echo "🔐 Using existing Hugging Face authentication..."
echo "   (If upload fails, run: hf auth login)"
echo ""

# Create the repository
echo "📦 Creating dataset repository: ${REPO_ID}..."
hf repo create ${REPO_NAME} --type dataset --exist-ok || {
    echo "⚠️  Repository might already exist, continuing..."
}
echo ""

# Create a temporary directory for upload
TEMP_DIR=$(mktemp -d)
echo "📁 Preparing files in temporary directory: ${TEMP_DIR}"

# Copy dataset files
echo "📋 Copying dataset files..."
cp "${DATASET_PATH}/dantes_conversations.jsonl" "${TEMP_DIR}/"
cp "${DATASET_PATH}/../citations/dantes.jsonl" "${TEMP_DIR}/dantes_citations.jsonl"

# Copy README
echo "📝 Copying documentation..."
cp "${DATASET_PATH}/DATASET_README.md" "${TEMP_DIR}/README.md"

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

hf upload ${REPO_ID} "${TEMP_DIR}" . --repo-type dataset

echo ""
echo "============================================================"
echo "✅ Upload complete!"
echo "============================================================"
echo "🔗 Your dataset is now available at:"
echo "   https://huggingface.co/datasets/${REPO_ID}"
echo ""
echo "📖 To use your dataset, run:"
echo ""
echo "   from datasets import load_dataset"
echo ""
echo "   # Load conversations (default)"
echo "   dataset = load_dataset(\"${REPO_ID}\")"
echo "   print(dataset['train'][0])"
echo ""
echo "   # Load raw citations"
echo "   citations = load_dataset(\"${REPO_ID}\", \"citations\")"
echo "   print(citations['train'][0])"
echo ""

# Clean up
rm -rf "${TEMP_DIR}"
echo "🧹 Cleaned up temporary files"
echo ""
echo "🎉 All done!"
