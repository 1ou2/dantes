import json
import re

def parse_instructions_file(input_file, output_file):
    """
    Parse instructions-result.txt which has multiple JSON objects per line
    and convert to proper JSONL format with ShareGPT-style conversations.
    """
    all_conversations = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # Find all JSON objects in the line using regex
            # Match {...} pattern, handling nested structures
            json_objects = []
            depth = 0
            start = None

            for i, char in enumerate(line):
                if char == '{':
                    if depth == 0:
                        start = i
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0 and start is not None:
                        json_str = line[start:i+1]
                        try:
                            obj = json.loads(json_str)
                            json_objects.append(obj)
                        except json.JSONDecodeError as e:
                            print(f"Line {line_num}: Failed to parse JSON: {e}")
                            print(f"  Problematic string: {json_str[:100]}...")
                        start = None

            # Convert each instruction/response pair to ShareGPT format
            for obj in json_objects:
                if 'instruction' in obj and 'response' in obj:
                    conversation = {
                        "conversations": [
                            {"from": "human", "value": obj["instruction"]},
                            {"from": "gpt", "value": obj["response"]}
                        ]
                    }
                    all_conversations.append(conversation)

    # Write to output file in JSONL format
    with open(output_file, 'w', encoding='utf-8') as f:
        for conv in all_conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')

    print(f"✓ Parsed {len(all_conversations)} instruction/response pairs")
    print(f"✓ Saved to {output_file}")

    return len(all_conversations)

def create_system_prompt_dataset(input_file, output_file, system_prompt):
    """
    Create a dataset with a custom system prompt for Edmond Dantès character.
    """
    all_conversations = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                # Add system message to the conversation
                conversation = {
                    "conversations": [
                        {"from": "system", "value": system_prompt},
                        {"from": "human", "value": data["conversations"][0]["value"]},
                        {"from": "gpt", "value": data["conversations"][1]["value"]}
                    ]
                }
                all_conversations.append(conversation)
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {e}")
                continue

    with open(output_file, 'w', encoding='utf-8') as f:
        for conv in all_conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')

    print(f"✓ Created dataset with system prompt: {len(all_conversations)} conversations")
    print(f"✓ Saved to {output_file}")

if __name__ == "__main__":
    import sys
    # Parse the raw instructions file
    input_file = sys.argv[1] if len(sys.argv) > 1 else "instructions-sample.txt"
    output_file = "data/dataset/dantes_conversations.jsonl"
    output_with_system = "data/dataset/dantes_conversations_system.jsonl"

    import os
    os.makedirs("data/dataset", exist_ok=True)

    # Step 1: Parse and convert to ShareGPT format
    count = parse_instructions_file(input_file, output_file)

    # Step 2: Create version with system prompt
    system_prompt = """Tu es Edmond Dantès, le Comte de Monte-Cristo. Tu t'exprimes avec le style et la dignité du XIXe siècle français. Ton ton est soutenu, élégant et empreint de mystère. Tu utilises des métaphores raffinées et maintiens une politesse formelle tout en gardant un contrôle absolu de la conversation."""

    create_system_prompt_dataset(output_file, output_with_system, system_prompt)

    print(f"\n✓ Dataset preparation complete!")
    print(f"  - Basic format: {output_file}")
    print(f"  - With system prompt: {output_with_system}")
