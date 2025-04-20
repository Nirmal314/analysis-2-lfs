import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

# Load your CSV
df = pd.read_csv("data/cleaned_ds.csv")

# Define keyword lists
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
    "botanist", "zoologist", "veterinarian", "pilot", "astronaut", "engineer"
]

# Labeling function
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

print("Label distribution:")
print(df["label"].value_counts())

# Split into train and test
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Tokenizer setup
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

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cpu":
    print("Warning: Training on CPU may be slow. Consider using a GPU if available.")

# Model setup
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

# Compute class weights for imbalance
class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=train_df['label'])
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print(f"Class weights: {class_weights}")

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=14)
eval_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=14)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 4
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# Custom training loop
best_accuracy = 0
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Training loss: {avg_train_loss:.4f}")
    
    # Evaluation
    model.eval()
    eval_loss = 0
    preds = []
    true_labels = []
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f"Evaluating Epoch {epoch+1}"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
            labels = batch['label'].to(device)
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fn(logits, labels)
            eval_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            preds.extend(predictions.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    avg_eval_loss = eval_loss / len(eval_loader)
    accuracy = accuracy_score(true_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average='binary')
    print(f"Epoch {epoch+1}, Evaluation loss: {avg_eval_loss:.4f}, Accuracy: {accuracy:.4f}, "
          f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    
    # Save best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"New best model saved with accuracy: {best_accuracy:.4f}")

# Load best model for inference
model.load_state_dict(torch.load('best_model.pth'))
print(f"Loaded best model with accuracy: {best_accuracy:.4f}")

# Prediction function
def predict_plot(plot_text):
    model.eval()
    inputs = tokenizer(plot_text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred = torch.argmax(probs, dim=-1).item()
    return "Major Stream" if pred == 0 else "Less Known Career"

# Example predictions
print(predict_plot("A young engineer struggles to build a startup."))
print(predict_plot("A physicist discovers a new particle."))