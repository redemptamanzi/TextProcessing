from transformers import pipeline

# Initialize the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define your categories with their definitions
labels = ["Technology and Innovation", "Health and Wellness", "Financial and Economic"]

# Example paragraphs to classify
paragraphs = [
    "This new AI model is revolutionizing the tech industry by improving the accuracy of predictions.",
    "Eating a balanced diet and exercising regularly are key to maintaining good health.",
    "The stock market experienced a significant drop due to economic instability."
]

# Perform zero-shot classification for each paragraph
for paragraph in paragraphs:
    result = classifier(paragraph, labels)
    print(f"Paragraph: {paragraph}")
    print(f"Predicted Category: {result['labels'][0]} with confidence {result['scores'][0]:.4f}")
    print("="*60)
