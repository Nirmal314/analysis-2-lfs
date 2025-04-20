import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load your CSV
df = pd.read_csv("data/cleaned_ds.csv")

major_keywords = [
    "engineer", "engineering", "doctor", "medical", "medicine", "hospital", "nurse",
    "lawyer", "law", "legal", "attorney", "judge", "court", "college", "university",
    "student", "professor", "teacher", "education", "school", "mba", "accountant",
    "chartered accountant", "ca", "civil services", "ias", "ips", "civil engineer",
    "mechanical engineer", "electrical engineer", "software engineer", "it professional",
    "architect", "pilot", "pharmacy", "pharmacist", "dentist", "veterinarian", "veterinary",
    "paramedic", "surgeon", "physician", "psychiatrist", "psychologist", "law enforcement",
    "police", "firefighter", "army", "navy", "air force"
]

less_known_keywords = [
    "researcher", "research", "scientist", "physicist", "chemist", "biologist",
    "mathematician", "astronomer", "geologist", "anthropologist", "archaeologist",
    "sportsman", "athlete", "cricketer", "footballer", "runner", "swimmer",
    "inventor", "artist", "musician", "dancer", "writer", "author", "poet",
    "filmmaker", "director", "photographer", "designer", "animator", "pilot",
    "explorer", "environmentalist", "activist", "philosopher", "historian",
    "librarian", "translator", "journalist", "chef", "gardener", "farmer",
    "botanist", "zoologist", "veterinarian", "astronaut"
]

def label_plot(plot):
    plot_lower = str(plot).lower()
    if any(word in plot_lower for word in major_keywords):
        return 0  # Major stream
    elif any(word in plot_lower for word in less_known_keywords):
        return 1  # Less known career
    else:
        return -1  # Unknown / unlabeled

df["label"] = df["plot"].apply(label_plot)

# Filter out unlabeled rows
df = df[df["label"] != -1]

print(df["label"].value_counts())

# df.to_csv("data/labeled_dataset.csv", index=False)

# Split into train and test
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(
        examples["plot"],
        padding="max_length",
        truncation=True,
        max_length=256,
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cpu":
    print("Warning: Training on CPU may be slow. Consider using a GPU if available.")

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

training_args = TrainingArguments(
    output_dir="./result",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
    dataloader_num_workers=14,  # Use all CPU cores for data loading
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

def predict_plot(plot_text):
    inputs = tokenizer(plot_text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred = torch.argmax(probs, dim=-1).item()
    return "Major Stream" if pred == 0 else "Less Known Career"

def load_model_and_predict(plot_text, model_path="./v1", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    inputs = tokenizer(
        plot_text, return_tensors="pt", padding=True, truncation=True, max_length=256
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()

    return "Major Stream" if pred == 0 else "Less Known Career"

if __name__ == '__main__':
    trainer.train()
    eval_results = trainer.evaluate()
    print(eval_results)

    # Save model and tokenizer
    model_save_path = "./v1"
    tokenizer_save_path = "./v1"
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(tokenizer_save_path)
    print(f"Model and tokenizer saved to {model_save_path}")

    # Example predictions using the loaded model function
    print(load_model_and_predict("A young engineer struggles to build a startup."))
    print(load_model_and_predict("A physicist discovers a new particle."))

# result: 

# {
#     'eval_loss': 0.28236472606658936, 
#     'eval_accuracy': 0.9216269841269841, 
#     'eval_f1': 0.5989847715736041, 
#     'eval_precision': 0.5841584158415841, 
#     'eval_recall': 0.6145833333333334, 
#     'eval_runtime': 174.8742, 
#     'eval_samples_per_second': 5.764, 
#     'eval_steps_per_second': 0.36, 
#     'epoch': 4.0
# }