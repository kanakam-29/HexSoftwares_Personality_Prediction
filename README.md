# HexSoftwares_Personality_Prediction
import nltk # type: ignore
nltk.download('stopwords')

import pandas as pd # type: ignore
import numpy as np # type: ignore
import re
import nltk # type: ignore
from nltk.corpus import stopwords # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.multioutput import MultiOutputClassifier # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.metrics import classification_report # type: ignore

# Download stopwords (run once)
nltk.download('stopwords')

# -----------------------------
# Sample Dataset (You can replace with your own CSV)
# Columns: resume_text + Big Five traits (0/1 or scores)
# -----------------------------
data = {
    "resume_text": [
        "Led multiple projects and collaborated with teams effectively",
        "Worked independently on research and analysis tasks",
        "Organized workflows and maintained documentation",
        "Creative thinker with strong design and innovation skills",
        "Handled client communication and team coordination"
    ],
    "openness": [1, 1, 0, 1, 0],
    "conscientiousness": [1, 1, 1, 0, 1],
    "extraversion": [1, 0, 0, 0, 1],
    "agreeableness": [1, 0, 1, 0, 1],
    "emotional_stability": [1, 1, 1, 1, 0]
}

df = pd.DataFrame(data)

# -----------------------------
# Text Preprocessing Function
# -----------------------------
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df["clean_text"] = df["resume_text"].apply(preprocess)

# -----------------------------
# Feature Extraction (TF-IDF)
# -----------------------------
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df["clean_text"])

# Target labels
y = df[[
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "emotional_stability"
]]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Model (Multi-label Classification)
# -----------------------------
model = MultiOutputClassifier(LogisticRegression())
model.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = model.predict(X_test)
print("Model Evaluation:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# Prediction Function
# -----------------------------
def predict_personality(resume_text):
    clean = preprocess(resume_text)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]

    traits = {
        "Openness": pred[0],
        "Conscientiousness": pred[1],
        "Extraversion": pred[2],
        "Agreeableness": pred[3],
        "Emotional Stability": pred[4]
    }

    return traits

# -----------------------------
# Example Usage
# -----------------------------
sample_resume = """
I worked on multiple team projects, led development teams,
and communicated with clients effectively. I enjoy innovation and problem solving.
"""

result = predict_personality(sample_resume)

print("\nPredicted Personality Traits:\n")
for trait, value in result.items():
    print(f"{trait}: {'High' if value == 1 else 'Low'}")
