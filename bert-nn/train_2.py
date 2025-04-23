import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizerFast, TFBertForSequenceClassification
import tensorflow as tf

# Load data
df = pd.read_csv("indian_movies_descriptions_ollama_categorized.csv")
df = df.dropna(subset=['description', 'career_category'])

texts = df['description'].astype(str).tolist()
labels = df['career_category'].tolist()

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

# Train/test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels_encoded, test_size=0.2, stratify=labels_encoded, random_state=42
)

# Load tokenizer and encode texts
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=256)

# Convert to tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels
)).shuffle(1000).batch(16)

val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    val_labels
)).batch(16)

# Load pretrained BERT model for sequence classification
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)

# Compile model
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = ['accuracy']

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Train model
history = model.fit(train_dataset, validation_data=val_dataset, epochs=3)

# Evaluate on validation set
loss, accuracy = model.evaluate(val_dataset)
print(f"Validation Accuracy: {accuracy:.4f}")

# Save model and tokenizer
model.save_pretrained('./bert_career_classifier')
tokenizer.save_pretrained('./bert_career_classifier')
