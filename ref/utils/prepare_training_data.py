from datasets import load_dataset
import json
from pathlib import Path
import ast


def save_json(split, out_path, english_col, hinglish_col):
    out_path.parent.mkdir(parents = True, exist_ok = True)

    items = []
    for row in split:
        input = row[english_col].strip()
        try:
            outputs = ast.literal_eval(row[hinglish_col].strip())
        except:
            outputs = row[hinglish_col].strip()
        
        if isinstance(outputs, (list, tuple)):
            outputs = [str(i) for i in outputs]
        else:
            outputs = [str(outputs)]

        for output in outputs:
            item = {
                "input": input,
                "output": output.strip()
            }
            items.append(item)

    with open(out_path, "w", encoding = "utf-8") as f:
        json.dump(items, f, ensure_ascii = False, indent = 2)

    print(f"Saved {len(items)} Items At : {out_path}")
    
def download_hinge_dataset_iitgn():
    dataset = load_dataset("LingoIITGN/HinGE")
    print(dataset["train"][:5])
    data_directory = Path("data")
    data_directory.mkdir(exist_ok = True)

    # Save splits into data/
    if "train" in dataset:
        print("Saving train split...")
        save_json(dataset["train"], data_directory / "hinge_train.json", "English", "Human-generated Hinglish")

    if "validation" in dataset:
        print("Saving validation split...")
        save_json(dataset["validation"], data_directory / "hinge_validation.json", "English", "Human-generated Hinglish")

    if "test" in dataset:
        print("Saving test split...")
        save_json(dataset["test"], data_directory / "hinge_test.json", "English", "Human-generated Hinglish")

    print("All files stored in the data/ folder.")

def download_hinge_dataset_findnitai():
    dataset = load_dataset("findnitai/english-to-hinglish")

    data_directory = Path("data")
    data_directory.mkdir(exist_ok = True)

    # Save splits into data/
    if "train" in dataset:
        print("Saving train split...")
        save_json(dataset["train"]['translation'], data_directory / "eng_hinglish_train.json", "en", "hi_ng")

    if "validation" in dataset:
        print("Saving validation split...")
        save_json(dataset["validation"], data_directory / "eng_hinglish_validation.json", "en", "hi_ng")

    if "test" in dataset:
        print("Saving test split...")
        save_json(dataset["test"], data_directory / "eng_hinglish_test.json", "en", "hi_ng")

    print("All files stored in the data/ folder.")


if __name__ == "__main__":
    download_hinge_dataset_iitgn()
    download_hinge_dataset_findnitai()