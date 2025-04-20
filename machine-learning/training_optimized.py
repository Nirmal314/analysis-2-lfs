import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import os
import multiprocessing

# Set environment variables for performance
os.environ["TOKENIZERS_PARALLELISM"] = "true"  # Enable parallel tokenization
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster matrix ops on Ampere GPUs
torch.backends.cudnn.benchmark = True  # Optimize CuDNN for dynamic input shapes

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

# Split into train and test
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

# Convert to Hugging Face Dataset and select only necessary columns
train_dataset = Dataset.from_pandas(train_df[["plot", "label"]])
test_dataset = Dataset.from_pandas(test_df[["plot", "label"]])

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(
        examples["plot"],
        padding="max_length",
        truncation=True,
        max_length=128,  # Reduced for faster processing
    )

# Tokenize datasets without caching
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Remove unnecessary columns and set format for PyTorch
train_dataset = train_dataset.remove_columns(["plot"])  # Remove 'plot' after tokenization
test_dataset = test_dataset.remove_columns(["plot"])
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cpu":
    print("Warning: Training on CPU may be slow. GPU is recommended for optimal performance.")

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Compile model for faster execution (PyTorch 2.0+)
if torch.__version__ >= "2.0":
    model = torch.compile(model)

num_workers = min(multiprocessing.cpu_count(), 14)
print(f"Using {num_workers} CPU cores for data loading")

training_args = TrainingArguments(
    output_dir="./result",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,  # Increased for GPU utilization
    per_device_eval_batch_size=32,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=torch.cuda.is_available(),  # Mixed precision training
    tf32=torch.cuda.is_available(),  # TensorFloat-32 for Ampere GPUs
    gradient_accumulation_steps=2,  # Accumulate gradients for larger effective batch size
    dataloader_num_workers=num_workers,
    dataloader_pin_memory=True,  # Faster data transfer to GPU
    dataloader_drop_last=False,  # Use all samples
    torch_compile=torch.__version__ >= "2.0",  # Compile model if PyTorch 2.0+
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
    inputs = tokenizer(plot_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():  # Disable gradient computation for inference
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred = torch.argmax(probs, dim=-1).item()
    return "Major Stream" if pred == 0 else "Less Known Career"

if __name__ == '__main__':
    trainer.train()
    eval_results = trainer.evaluate()
    print(eval_results)

    # Example predictions
    print(predict_plot("A young engineer struggles to build a startup."))
    print(predict_plot("A physicist discovers a new particle."))