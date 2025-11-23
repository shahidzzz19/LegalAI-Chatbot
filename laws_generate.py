import json
import re

def txt_to_json(txt_file, json_file):
    data = {"IPC": {}}

    with open(txt_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Pattern to capture: [s section_number] Title\nContent
    # Example: [s 379.6.14] Motor vehicles\nThe allegation...
    pattern = re.compile(r"\[s ([\d\.]+)\]\s*(.*?)\n(.*?)(?=\n\[s|\Z)", re.S)

    matches = pattern.findall(content)

    for section, title, body in matches:
        data["IPC"][f"section{section}"] = {
            "title": title.strip(),
            "content": body.strip()
        }

    # Save JSON
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Converted {len(matches)} sections into {json_file}")

# Example usage
txt_to_json("data\data_ipc_law.txt", "laws_raw.json")
