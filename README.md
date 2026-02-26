# 🐶 Dog Breed Identification using Transfer Learning

This project uses MobileNetV2 and deep learning to identify dog breeds from images.

## How to Run

1. Install dependencies
   pip install -r requirements.txt
2. Run the app
   streamlit run app.py
   
## Model Limitation

The model is trained only on dog breed datasets.
When a non-dog image is uploaded, the model still predicts
the closest matching dog breed due to closed-set classification.