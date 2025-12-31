# dantes
Finetuning a model to act as Edmond Dantès.
Target model is mistral-7b-instruct.

# Install
uv venv
source .venv/bin/activate
#uv pip install -r requirements.txt
./setup.sh

# RUN Finetunig
python finetune_dantes.py

If you have an error : Error Internal Triton PTX codegen error
`ptxas` stderr:
ptxas fatal   : Value 'sm_121a' is not defined for option 'gpu-name'
The solution is 
```bash
export TORCH_CUDA_ARCH_LIST=12.1a
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
python finetune_dantes.py
```

# Gutenberg file processing
File : `src/gutenberg.py`
- download the 4 files corresponding to the original french text
- remove UTF-8 BOM
- remove header and footer
- reformat text and remove unecessary line breaks around column 80

Output of this file is a clean formatted file in the data/gutenberg folder

# Hugging fate
## install hugging face cli tool
curl -LsSf https://hf.co/cli/install.sh | bash

## create an access token
Go to your hugging face account 

## Download
How to download a model ? Example :
```bash
hf auth login
hf download mistralai/Ministral-3-3B-Instruct-2512
ls -al $HOME/.cache/huggingface/hub/models--mistralai--Ministral-3-3B-Instruct-2512/snapshots/811c44083b80c8026885759afdece4413740632f/
total 24
drwxrwxr-x 2 gabriel gabriel 4096 déc.  21 11:37 .
drwxrwxr-x 3 gabriel gabriel 4096 déc.  21 11:36 ..
lrwxrwxrwx 1 gabriel gabriel   52 déc.  21 11:36 chat_template.jinja -> ../../blobs/32d54c07b1f7ee6e50d5b60f8b112e6e94abc583
lrwxrwxrwx 1 gabriel gabriel   52 déc.  21 11:36 config.json -> ../../blobs/dc2540f4f6c00e131b2f29dc7e7f5fc2805c8f41
lrwxrwxrwx 1 gabriel gabriel   76 déc.  21 11:37 consolidated.safetensors -> ../../blobs/a1b2aa6d22874ed04b7071a595581d48832ddeda8d6e69f54537b31aa3a775cf
lrwxrwxrwx 1 gabriel gabriel   52 déc.  21 11:36 generation_config.json -> ../../blobs/add11cbc06647495098ee6dd5c9cbc96841a445a
lrwxrwxrwx 1 gabriel gabriel   52 déc.  21 11:36 .gitattributes -> ../../blobs/0d4cb185280917cac60ef7195f2a6250b2b90d83
lrwxrwxrwx 1 gabriel gabriel   76 déc.  21 11:37 model.safetensors -> ../../blobs/728f1826cd0e38191ca7b1379e81f78cf0555c6ffd95882aabd2404632346f86
lrwxrwxrwx 1 gabriel gabriel   52 déc.  21 11:36 params.json -> ../../blobs/00cb7057abcfd2874d8eba5cc72eb2d84d8e3bdc
lrwxrwxrwx 1 gabriel gabriel   52 déc.  21 11:36 processor_config.json -> ../../blobs/a37d728b12fd27ac60a437894bd51de83449bf30
lrwxrwxrwx 1 gabriel gabriel   52 déc.  21 11:36 README.md -> ../../blobs/bc1e39adcd1f7b30292a802dcd59d39c76d24175
lrwxrwxrwx 1 gabriel gabriel   52 déc.  21 11:36 SYSTEM_PROMPT.txt -> ../../blobs/5f95aa977b9fefcf6fb3a683fa97779a7f2c0e93
lrwxrwxrwx 1 gabriel gabriel   76 déc.  21 11:36 tekken.json -> ../../blobs/e29d19ea32eb7e26e6c0572d57cb7f9eca0f4420e0e0fe6ae1cf3be94da1c0d6
lrwxrwxrwx 1 gabriel gabriel   52 déc.  21 11:36 tokenizer_config.json -> ../../blobs/a7843c180f2b39d43303e7eba55d2e34fd600a8f
lrwxrwxrwx 1 gabriel gabriel   76 déc.  21 11:36 tokenizer.json -> ../../blobs/286acad9b0e27fce778ac429763536accf618ccb6ed72963b6f94685e531c5c7
```

# Dataset

## Citation dataset
File: `src/citation.py`
Objective: Generate a dataset of phrases, thoughts, and dialogues from Edmond Dantès.

  To achieve this:
  - Split the text into chunks of 100 lines
  - Build a prompt asking to extract citations from the block of lines
  - Use llama.cpp locally (gpt-oss-20b model) listening on the /chat/completions endpoint
  - In Python, send a POST request to extract citations from these lines
  - Generate with the LLM a series of JSON objects: {"context":"citation context", "citation": "phrase spoken by Edmond Dantès"}
  - Save the citations in a jsonl file

  This describes a workflow for extracting character-specific quotes from "The Count of Monte Cristo" using a local LLM to analyze text chunks and generate structured citation data.

## Character card
File: `src/charactercard.py`
Based on the citations from Dantès, generate a character card description the tone, style and personality of Edmond Dantès

## instructions
File: `src/instructions.py`
Analyse citations and create a set of questions and answers that will be used during the finetuning