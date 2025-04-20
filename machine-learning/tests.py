from transformers import AutoTokenizer
import torch
from transformers import AutoModelForSequenceClassification


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

    print(pred)

    return "Major Stream" if pred == 0 else "Unknown" if pred == -1 else "Less Known Career"

plot = input("Enter movie plot: ")
print(load_model_and_predict(plot))

# Dr. Freddy Ginwala is a shy, introverted dentist who lives a quiet and lonely life. Beneath his soft-spoken nature hides a dark, obsessive side. When he falls in love with a woman trapped in an abusive marriage, his obsession turns deadly. What starts as a love story soon spirals into suspense and murder, revealing Freddy’s deeply twisted mind.