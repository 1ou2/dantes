---
language:
- fr
license: cc-by-4.0
task_categories:
- text-generation
- conversational
size_categories:
- 1K<n<10K
tags:
- literature
- french
- character
- roleplay
- dumas
- monte-cristo
- synthetic
pretty_name: Edmond Dantès Conversation Dataset
dataset_info:
  - config_name: conversations
    features:
    - name: conversations
      list:
      - name: from
        dtype: string
      - name: value
        dtype: string
    splits:
    - name: train
      num_examples: 4091
  - config_name: citations
    features:
    - name: contexte
      dtype: string
    - name: citation
      dtype: string
    splits:
    - name: train
      num_examples: 1494
configs:
- config_name: conversations
  data_files:
  - split: train
    path: "dantes_conversations.jsonl"
- config_name: citations
  data_files:
  - split: train
    path: "dantes_citations.jsonl"
---

# Edmond Dantès Conversation Dataset

This dataset contains synthetic conversational data and source citations for fine-tuning language models to embody the character of Edmond Dantès from Alexandre Dumas' classic novel "Le Comte de Monte-Cristo" (The Count of Monte Cristo). The conversations are in formal 19th-century French, maintaining the literary style and personality of the protagonist.

The dataset includes two configurations:
- **`conversations`** (default): 4,091 instruction-response pairs for fine-tuning
- **`citations`**: 1,494 raw citations extracted from the source text

## Dataset Description

### Dataset Summary

This dataset was created through a multi-stage pipeline:
1. **Source text extraction**: Downloaded and preprocessed 4 volumes of "Le Comte de Monte-Cristo" from Project Gutenberg (IDs: 17989, 17990, 17991, 17992)
2. **Citation extraction**: Used an LLM to extract Dantès' thoughts, speech, and writings from the original French text
3. **Synthetic conversation generation**: Generated question-answer pairs based on the extracted citations

The result is a high-quality dataset for training conversational AI models to respond as Edmond Dantès with authentic personality traits and 19th-century formal French style.

### Supported Tasks

- **Character Roleplay**: Training models to respond as a specific literary character
- **Conversational AI**: Instruction-tuned dialogue generation
- **French Language Models**: Fine-tuning on formal, literary French
- **Literary Analysis**: Understanding character voice and perspective

### Languages

- French (fr) - Original text from Alexandre Dumas
- Formal 19th-century literary style

## Dataset Structure

### Data Instances

#### Conversations Config (default)

Each instance contains a conversation in ShareGPT format:

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "Que pensez-vous de la justice et de la vengeance?"
    },
    {
      "from": "gpt",
      "value": "La justice, monsieur, est le fondement de toute société civilisée..."
    }
  ]
}
```

#### Citations Config

Each instance contains a raw citation extracted from the source text:

```json
{
  "contexte": "Dantès informe Morrel de la perte du capitaine Leclère lors de l'arrivée du Pharaon à Marseille",
  "citation": "--Un grand malheur, monsieur Morrel! répondit le jeune homme, un grand malheur, pour moi surtout: à la hauteur de Civita-Vecchia, nous avons perdu ce brave capitaine Leclère."
}
```

### Data Fields

#### Conversations Config
- `conversations`: List of conversation turns
  - `from`: Speaker role ("human" or "gpt")
  - `value`: Message content (string)

#### Citations Config
- `contexte`: Context describing the situation (string)
- `citation`: Actual quote from Dantès (dialogue, thought, or writing) (string)

### Data Splits

- **Conversations**: 4,091 instruction-response pairs
- **Citations**: 1,494 raw citations from the source text
- No validation/test splits (designed for full fine-tuning)

## Dataset Creation

### Source Data

**Original Text**: "Le Comte de Monte-Cristo" by Alexandre Dumas
- Volume 1: Project Gutenberg #17989
- Volume 2: Project Gutenberg #17990
- Volume 3: Project Gutenberg #17991
- Volume 4: Project Gutenberg #17992

**License**: Public domain (published 1844-1846)

### Data Collection Process

1. **Text Preprocessing** (`src/gutenberg.py`):
   - Downloaded source texts from Project Gutenberg
   - Removed UTF-8 BOM, headers, footers
   - Reformatted text: preserved paragraphs, fixed line breaks
   - Special handling for dialogue markers (—, --, –, -, «)

2. **Citation Extraction** (`src/citation.py`):
   - Split text into 100-line chunks with 10-line overlap
   - Used local LLM to extract Dantès' dialogue, thoughts, and writings
   - Output: JSONL with `contexte` and `citation` fields
   - Result: 1,494 citations (~337 KB)

3. **Instruction Generation** (`src/instructions.py`):
   - For each citation, generated 3 question-answer pairs
   - Maintained 19th-century formal style
   - Focused on character perspective and themes

4. **Dataset Formatting** (`src/prepare_dataset.py`):
   - Converted to ShareGPT conversation format
   - Final output: 4,091 instruction-response pairs

### Annotations

**Annotation Process**: Fully synthetic, generated by LLM with careful prompt engineering

**Annotators**: Automated using local LLM server (llama.cpp)

**Quality Control**:
- Manual review of citation extraction quality
- Validation of JSON formatting
- Character consistency checks in generated conversations

### Personal and Sensitive Information

None. All content is derived from a 180-year-old public domain novel.

## Considerations for Using the Data

### Social Impact

This dataset represents a fictional character from 19th-century French literature. Users should be aware of:

- **Historical context**: Reflects social norms and attitudes of 1840s France as portrayed in fiction
- **Character-specific viewpoints**: Embodies complex perspectives on justice, vengeance, and morality
- **Literary nature**: Not intended for factual information or modern ethical guidance

### Biases

**Content Biases**:
- Trained on a single literary work (limited perspective)
- Over-represents themes of revenge, justice, betrayal, and redemption
- Reflects 19th-century French aristocratic worldview

**Style Biases**:
- Formal, archaic French language
- Literary and philosophical tone
- May not reflect modern conversational patterns

### Limitations

- **Scope**: Limited to knowledge within "Le Comte de Monte-Cristo"
- **Language**: Formal French only, not suitable for modern casual conversation
- **Character-bound**: Designed for roleplay, not general-purpose tasks
- **Synthetic data**: Generated by LLM, may contain artifacts or inconsistencies

## Additional Information

### Dataset Curators

Created by Gabriel Pastor ([1ou2](https://huggingface.co/1ou2))

### Licensing Information

- **Dataset License**: CC-BY 4.0
- **Source Text**: Public domain (1844-1846)
- **Synthetic Content**: Generated content licensed under CC-BY 4.0

### Citation Information

If you use this dataset in your work, please cite:

```bibtex
@misc{dantes-dataset-2025,
  title={Edmond Dantès Conversation Dataset},
  author={Gabriel Pastor},
  year={2025},
  howpublished={\url{https://huggingface.co/datasets/1ou2/comte-monte-cristo-conversations}},
  note={Synthetic conversational dataset from "Le Comte de Monte-Cristo"}
}
```

### Source Text Citation

```bibtex
@book{dumas1844monte,
  title={Le Comte de Monte-Cristo},
  author={Dumas, Alexandre},
  year={1844-1846},
  publisher={Pétion},
  note={Available from Project Gutenberg}
}
```

### Related Resources

- **Fine-tuned Model**: [1ou2/comte-monte-cristo-mistral-7b](https://huggingface.co/1ou2/comte-monte-cristo-mistral-7b)
- **Source Code**: [github.com/1ou2/dantes](https://github.com/1ou2/dantes)
- **Project Gutenberg**: [gutenberg.org](https://www.gutenberg.org/)

### Contact

For questions or feedback:
- GitHub Issues: [github.com/1ou2/dantes/issues](https://github.com/1ou2/dantes/issues)
- Hugging Face: [@1ou2](https://huggingface.co/1ou2)

---

## Usage Example

### Loading the Dataset

```python
from datasets import load_dataset

# Load conversations (default)
dataset = load_dataset("1ou2/comte-monte-cristo-conversations")
print(dataset['train'][0])

# Load raw citations
citations = load_dataset("1ou2/comte-monte-cristo-conversations", "citations")
print(citations['train'][0])
```

### Using for Fine-tuning

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Load dataset
dataset = load_dataset("1ou2/comte-monte-cristo-conversations")

# Setup training
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset['train'],
    dataset_text_field="text",
    # ... your training config
)

trainer.train()
```

### Format Conversion

The dataset uses ShareGPT format, compatible with most instruction-tuning frameworks:

```python
# Convert to Alpaca format
def to_alpaca_format(example):
    return {
        "instruction": example['conversations'][0]['value'],
        "output": example['conversations'][1]['value']
    }

alpaca_dataset = dataset.map(to_alpaca_format)
```

### Dataset Statistics

**Conversations Config:**
- Total conversations: 4,091
- Average question length: ~50 tokens
- Average response length: ~150 tokens
- File size: 3.2 MB

**Citations Config:**
- Total citations: 1,494
- Average citation length: ~100 tokens
- File size: 337 KB

**Format**: JSONL (JSON Lines) for both configs
