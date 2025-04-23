import pandas as pd
import subprocess
import time

# Load descriptions
df = pd.read_csv("indian_movies_descriptions.csv")

def classify_career_with_ollama(description):
    prompt = f"""
You are an expert in career analysis. Given the following movie description, identify the main profession or career mentioned. 
Then classify it into one of these categories:
- Highly chosen career in India (1)
- Lesser chosen career in India (0)
- None (if no clear profession or unrelated) (-1)

Description: \"\"\"{description}\"\"\"

Respond ONLY with one of the categories exactly as above.
"""
    try:
        # Run ollama CLI command
        result = subprocess.run(
            ["ollama", "run", "llama3.2", prompt],
            capture_output=True,
            text=True,
            check=True,
        )
        # The output is in result.stdout
        category = result.stdout.strip()
        return category
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return "None"

# Apply AI classification to each description (with delay to avoid rate limits)
categories = []
for i, desc in enumerate(df['description']):
    print(f"Processing {i+1}/{len(df)}")
    category = classify_career_with_ollama(desc)
    print(f"Category: {category}\n")
    categories.append(category)
    time.sleep(1)  # adjust delay as needed

df['career_category'] = categories

# Save results
df.to_csv("indian_movies_descriptions_ollama_categorized.csv", index=False)

print("Ollama categorized descriptions saved to indian_movies_descriptions_ollama_categorized.csv")
