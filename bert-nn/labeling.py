import pandas as pd
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load descriptions
df = pd.read_csv("data/cleaned_ds.csv")

def classify_career_with_ollama(description, index):
    print(f"[{index}] Starting classification...")
    prompt = f"""
You are an expert in career analysis. Given the following movie description, identify the main profession or career mentioned in the text.

Then classify the identified profession into one of the following categories:
- "1" if it is a highly chosen career in India,
- "0" if it is a lesser chosen career in India,
- "-1" if there is no clear profession mentioned or if the description is unrelated.

Description:
\"\"\"{description}\"\"\"

Please respond with ONLY one of the following exact values: 1, 0, or -1.
No additional text or explanation is needed.
"""

    try:
        result = subprocess.run(
            ["ollama", "run", "gemma3", prompt],
            capture_output=True,
            text=True,
            check=True,
        )
        category = result.stdout.strip()
        print(f"[{index}] Completed: Category = {category}")
        return category
    except subprocess.CalledProcessError as e:
        print(f"[{index}] Error: {e}")
        return "-1"  # consistent with your categories

def main():
    descriptions = df['plot'].tolist()
    categories = [None] * len(descriptions)

    max_workers = 10  # Adjust based on your system capabilities

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(classify_career_with_ollama, desc, idx): idx
            for idx, desc in enumerate(descriptions)
        }

        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                categories[idx] = future.result()
            except Exception as exc:
                print(f"[{idx}] Exception: {exc}")
                categories[idx] = "-1"

    df['career_category'] = categories
    df.to_csv("data/result.csv", index=False)
    print("Ollama categorized descriptions saved to result.csv")

if __name__ == "__main__":
    main()
