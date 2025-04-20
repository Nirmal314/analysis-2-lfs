from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch"
)

print("It works ✅")
