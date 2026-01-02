---
base_model: unsloth/mistral-7b-instruct-v0.3-bnb-4bit
library_name: peft
model_name: Edmond Dantès - Le Comte de Monte-Cristo
tags:
- base_model:adapter:unsloth/mistral-7b-instruct-v0.3-bnb-4bit
- lora
- sft
- transformers
- trl
- unsloth
- character
- roleplay
- french
- literature
- dumas
license: apache-2.0
pipeline_tag: text-generation
language:
- fr
datasets:
- custom
---

# Edmond Dantès - Le Comte de Monte-Cristo

Fine-tuned Mistral 7B model trained to embody the character of Edmond Dantès from Alexandre Dumas' classic novel "Le Comte de Monte-Cristo" (The Count of Monte Cristo). This model responds in character, maintaining the formal 19th-century French style and personality of the protagonist.

## Model Description

This is a LoRA (Low-Rank Adaptation) fine-tuned version of [unsloth/mistral-7b-instruct-v0.3-bnb-4bit](https://huggingface.co/unsloth/mistral-7b-instruct-v0.3-bnb-4bit), trained using the Unsloth framework for efficient fine-tuning. The model was trained on a custom dataset of 4,514 instruction-response pairs extracted from the original French text of "Le Comte de Monte-Cristo" (Project Gutenberg).

### Key Features

- **Character consistency**: Responds as Edmond Dantès with authentic personality traits
- **19th-century formal French**: Maintains the literary style of the original novel
- **Literary knowledge**: Draws from events, relationships, and themes from the source material
- **Instruction-tuned**: Formatted for conversational interactions

## Training Details

### Dataset

- **Source**: "Le Comte de Monte-Cristo" by Alexandre Dumas (4 volumes from Project Gutenberg: IDs 17989, 17990, 17991, 17992)
- **Size**: 4,091 instruction-response conversation pairs
- **Format**: ShareGPT-style conversations with user questions and character responses
- **Language**: French (original text)
- **Extraction method**: Citations of Dantès' thoughts, speech, and writings were extracted from the source text and converted into instruction-tuning format

### Training Procedure

**Source code** : [Dantes](https://github.com/1ou2/dantes) 
**Framework**: [Unsloth](https://github.com/unslothai/unsloth) - optimized for fast and memory-efficient fine-tuning
**Base model**: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit` (4-bit quantized Mistral 7B Instruct v0.3)

**Training hyperparameters**:
- **LoRA configuration**:
  - Rank (r): 16
  - Alpha: 16
  - Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
  - Dropout: 0
  - Bias: none
- **Training parameters**:
  - Epochs: 3
  - Batch size per device: 8
  - Gradient accumulation steps: 1
  - Effective batch size: 8
  - Learning rate: 1.875e-4 (0.0001875)
  - Learning rate scheduler: linear
  - Warmup steps: 5
  - Optimizer: AdamW 8-bit
  - Weight decay: 0.001
  - Max sequence length: 2048
  - Packing: Enabled
- **Template**: ChatML format
- **Total training steps**: 1,536
- **Training time**: ~146.84 minutes (~2.45 hours) on NVIDIA DGX Spark (GB10 with 128 GB VRAM)

**Training metrics**:
- Average training loss: 0.9645
- Starting loss: ~2.24
- Final loss: ~0.58
- Train samples per second: 1.393
- Train steps per second: 0.174
- Peak GPU memory usage: 7.787 GB (6.506% of total)
- LoRA memory: 1.021 GB

### Framework Versions

- PEFT: 0.18.0
- TRL: 0.24.0
- Transformers: 4.57.3
- PyTorch: 2.9.1+cu130
- Datasets: 4.3.0
- Tokenizers: 0.22.1
- Unsloth: 2025.12.9
- CUDA: 12.1 / Toolkit: 13.0
- Triton: 3.5.1

## Usage

### Loading with Unsloth (Recommended)

```python
from unsloth import FastLanguageModel

# Load the model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="1ou2/comte-monte-cristo-mistral-7b",
    max_seq_length=2048,
    dtype=None,  # Auto-detect
    load_in_4bit=True,
)

# Enable inference mode
FastLanguageModel.for_inference(model)

# Prepare a conversation
messages = [
    {"role": "user", "content": "Que pensez-vous de la justice et de la vengeance?"}
]

# Apply chat template
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

# Generate response
outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

# Decode and print
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Loading with Transformers + PEFT

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    device_map="auto",
    load_in_4bit=True
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "1ou2/comte-monte-cristo-mistral-7b/")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("1ou2/comte-monte-cristo-mistral-7b/")

# Use the model
messages = [
    {"role": "user", "content": "Qui êtes-vous?"}
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

outputs = model.generate(inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Chat Template Format

The model uses ChatML format:

```
<|im_start|>user
Votre question ici<|im_end|>
<|im_start|>assistant
Réponse du modèle<|im_end|>
```

## Example Interactions

**Question**: "Qui es-tu ?"

**Response**: "Je suis le comte de Monte-Cristo, un homme dont le passé est un labyrinthe de trahisons et de vengeance, et dont le présent se tisse à la fois dans la lumière des rédemptions et dans l’ombre des secrets. Mon identité, comme un phare dans la nuit, guide ceux qui cherchent la justice, mais elle ne se révèle qu’aux âmes capables de percevoir l’écho des injustices passées. Je suis l’ombre qui se prolonge sur les murs de la vérité, et je suis la lumière qui perce les ténèbres de l’injustice. Mon nom, Monte-Cristo, est le symbole d’une quête éternelle, où chaque pas est un pas vers la réconciliation du monde."

---

**Question**: "Qu'est-ce que le fine-tuning"

**Response**: "Le fine-tuning, cher interlocuteur, est l'art de polir les arêtes de l'intelligence artificielle, de façon à lui conférer la délicatesse d'un poème bien versé. Il consiste à ajuster les paramètres de l'algorithme, comme on ajuste les cordes d'un violon, afin que la musique qu'il produit soit à la fois puissante et douce. Cette opération, loin d'être un simple ajustement, est une danse subtile entre la précision et la fluidité, où chaque nuance compte. C'est ainsi que l'on transforme un simple outil en un compagnon de pensée, capable de suivre les tourbillons de la conscience humaine sans perdre son équilibre."


## Limitations and Biases

### Limitations

- **Character-specific**: This model is designed to respond as Edmond Dantès and may not be suitable for general-purpose tasks
- **19th-century language**: Uses formal, archaic French that may not reflect modern conversational style
- **Source text bounded**: Knowledge is limited to events, characters, and themes from "Le Comte de Monte-Cristo"
- **LoRA adapter**: Requires the base Mistral 7B Instruct v0.3 model to function
- **Language**: Primarily trained on French text

### Biases

- **Literary character perspective**: Embodies the worldview, values, and personality of a fictional 19th-century character
- **Historical context**: Reflects social norms, attitudes, and language of 1840s France as portrayed in the novel
- **Dataset bias**: Trained on citations from a single literary work, which may over-represent certain themes (revenge, justice, betrayal, redemption)

## Intended Use

### Primary Use Cases

- **Creative writing**: Assistance with writing fanfiction or stories in the Monte Cristo universe
- **Educational**: Learning about the character and the novel through interactive dialogue
- **Literary analysis**: Exploring character motivations and perspectives
- **Roleplay**: Interactive storytelling and character roleplay
- **French language practice**: Practicing formal, literary French

### Out-of-Scope Use

- General-purpose question answering
- Factual information retrieval (beyond the novel's content)
- Modern conversational French assistance
- Any use requiring breaking character

## Ethical Considerations

This model represents a fictional character from a 19th-century novel. Users should be aware that:

- The character's views on justice, vengeance, and morality are complex and context-dependent
- Historical attitudes and social norms reflected in the source material may not align with contemporary values
- This is a creative/educational tool, not a source of advice or factual information

## Citation

If you use this model in your work, please cite:

```bibtex
@misc{dantes-lora-2025,
  title={Edmond Dantès: Fine-tuned Mistral 7B for Character Roleplay},
  author={[Gabriel Pastor]},
  year={2025},
  howpublished={\url{https://huggingface.co/1ou2/comte-monte-cristo-mistral-7b/}},
  note={LoRA fine-tuning of Mistral 7B Instruct v0.3 on "Le Comte de Monte-Cristo"}
}
```

### Acknowledgments

- **Base model**: [Mistral AI](https://mistral.ai/) for Mistral 7B Instruct v0.3
- **Optimization**: [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning framework
- **Source text**: Alexandre Dumas' "Le Comte de Monte-Cristo" via [Project Gutenberg](https://www.gutenberg.org/)
- **Framework**: [Hugging Face](https://huggingface.co/) Transformers, PEFT, and TRL libraries

### Framework Citations

**TRL (Transformer Reinforcement Learning)**:

```bibtex
@misc{vonwerra2022trl,
  title        = {{TRL: Transformer Reinforcement Learning}},
  author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeking and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
  year         = 2020,
  journal      = {GitHub repository},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/huggingface/trl}}
}
```

**Unsloth**:

```bibtex
@misc{unsloth2024,
  title={Unsloth: Fast and memory-efficient LLM fine-tuning},
  author={Unsloth AI},
  year={2024},
  howpublished={\url{https://github.com/unslothai/unsloth}}
}
```

## License

This model adapter is released under the Apache 2.0 license. The base Mistral 7B Instruct v0.3 model has its own license terms. The source text "Le Comte de Monte-Cristo" is in the public domain.

## Model Card Contact

For questions or feedback about this model, please open an issue in the [source repository](https://github.com/1ou2/dantes) or contact via Hugging Face.
