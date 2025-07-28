Fake vs. Real News Classification using NLP and Machine Learning
Project Summary
This project aims to identify whether a news article is genuine or fabricated using Natural Language Processing (NLP) and a Logistic Regression model. It was implemented as part of the project phase during the Elevate Labs Internship.

Datasets Utilized
True.csv.xlsx: Contains authentic news articles

Fake.csv.xlsx: Contains fabricated news articles

The datasets were merged into a single dataset and preprocessed for model training and evaluation.

Tools and Technologies
Programming Language: Python

Libraries: scikit-learn, pandas, NumPy

Text Processing: NLTK

Vectorization: TF-IDF (Top 5000 features)

Model: Logistic Regression

Model Overview
Preprocessing Techniques:

Convert to lowercase

Remove stopwords

Apply stemming

Feature Extraction: TF-IDF vectorizer

Classifier Used: Logistic Regression

Test Accuracy: ~98.9%

How to Run Predictions
Example: Predicting Custom News Article
python
Copy
Edit
text = "The government launches a new digital health mission."
cleaned = preprocess(text)
vector = tfidf.transform([cleaned])
model.predict(vector)
Included Files
File Name	Description
news_model.pkl	Trained logistic regression model
vectorizer.pkl	TF-IDF transformer object
News_Classification_Project_Output.pdf	Summary of project steps and output
True.csv.xlsx & Fake.csv.xlsx	Raw datasets

Sample Prediction
Input: "The government launches a new health mission"
Prediction: REAL

(Optional) Streamlit App
To run the application interface using Streamlit:

bash
Copy
Edit
streamlit run app.py
Author
Jeon Jiju
Intern, Elevate Labs (AI/ML Track)
