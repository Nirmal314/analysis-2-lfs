from transformers import BertTokenizerFast, TFBertForSequenceClassification
import tensorflow as tf
import numpy as np

# Load the saved model and tokenizer once
tokenizer = BertTokenizerFast.from_pretrained('./bert_career_classifier')
model = TFBertForSequenceClassification.from_pretrained('./bert_career_classifier')

# Load label encoder classes (adjust if you saved it differently)
# For example, if you saved label encoder classes as a list:
label_classes = [
    "Highly chosen career",
    "Lesser chosen career",
    "None"
]

def predict_career_category(plot: str) -> str:
    """
    Predict the career category from a movie plot description.

    Args:
        plot (str): The movie description text.

    Returns:
        str: Predicted career category.
    """
    # Tokenize input text
    inputs = tokenizer(plot, return_tensors="tf", truncation=True, padding=True, max_length=256)

    # Get model outputs (logits)
    outputs = model(inputs)

    logits = outputs.logits
    predicted_class_id = tf.argmax(logits, axis=1).numpy()[0]

    # Map predicted class id to label
    predicted_label = label_classes[predicted_class_id]

    return predicted_label


# Example usage:
if __name__ == "__main__":
    # sample_plot = (
    #     "Dangal is a biography of a real life patriotic fighter Mahavir Singh Phogat who raises his daughters and evolves them into World Class Fighters. "
    #     "The movie begins with a crochy brawl between Mahavir (Aamir Khan) and his colleague who was also former wrestler, thus embalming the terrific Dangal Theme, discovering Mahavir's past life of a Wrestler. "
    #     "Mahavir admires to make his dreams come true by his sons but on contrary, four daughters take birth. "
    #     "They come with a complaint of beating down a boy in their locality. "
    #     "Hence Mahavir hopes they would be future Wrestlers. "
    #     "So he trains his daughters Gita and Babita, thus proving they're no less than a professional male wrestler. "
    #     "After a long struggle, Gita enters the National level Wrestling and acquires victory. "
    #     "When she is up to the Internationals, she opts to get trained from the NSA(National Sports Academy) where a coach mis-trains Gita, due to which she deliberately fails every match she attempts. "
    #     "Her sister Babita also attains an age to get into Wresting. Now, Mahavir plans to train both by his own norms. Finally, Gita defeats an Australian Wrestler by following the predominant path of her father, not of the coach. The movie is par excellent, enridges Women Empowerment in the Nation. "
    # )
    
    plots = [
        """A talented teenage singer/songwriter living amid domestic abuse becomes a YouTube sensation after a video in which she hides her identity goes viral. A talented teenage singer/songwriter living amid domestic abuse becomes a YouTube sensation after a video in which she hides her identity goes viral.""",
        """The story of courageous Neerja Bhanot a flight attendant, who sacrificed her life while protecting the lives of 359 passengers on Pan Am Flight 73 in 1986 when it was hijacked by a terrorist organization.""",
        """Ishaan, a student who has dyslexia, cannot seem to get anything right at his boarding school. Soon, a new unconventional art teacher, Ram Shankar Nikumbh, helps him discover his hidden potential. Content collapsed""",
        """The film is an adaptation of Harinder Sikka's 2008 novel Calling Sehmat, a true account of an Indian Research and Analysis Wing (RAW) agent who, upon her father's request, is married into a family of military officers in Pakistan to relay information to India, prior to the Indo-Pakistani War of 1971.
        """
    ]
    
    for plot in plots:
        prediction = predict_career_category(plot)
        print(f"Plot: {plot}")
        print(f"Predicted career category: {prediction}\n\n")
        