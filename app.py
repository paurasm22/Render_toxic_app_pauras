import streamlit as st
from transformers import pipeline
import pandas as pd
import re

@st.cache_resource
def load_model():
    model_path = "Pau22/distilbert-toxic-model"
    return pipeline(
        "text-classification",
        model=model_path,
        tokenizer=model_path,
        return_all_scores=False,
    )

classifier = load_model()

def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", str(text))
    text = text.encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", text).strip()

LABEL_MAP = {"LABEL_0": "Not Toxic", "LABEL_1": "Toxic"}

st.set_page_config(page_title="Toxic Comment Classifier", layout="centered")
st.title("🧠 Toxic Comment Classifier — DistilBERT by Pau22")
st.write("Fine-tuned on the Jigsaw Toxic Comment dataset.")

toxic_samples = [
    "You are the worst person ever.",
    "Shut up you idiot.",
    "You f__king clown.",
    "Nobody likes you, go away.",
]

non_toxic_samples = [
    "Have a lovely day!",
    "Thank you for your help!",
    "I appreciate your effort.",
    "This was very helpful, thanks!",
]

col1, col2 = st.columns(2)

with col1:
    toxic_choice = st.selectbox("Choose a Toxic Example (Optional)", ["-- None --"] + toxic_samples)

with col2:
    non_toxic_choice = st.selectbox("Choose a Non-Toxic Example (Optional)", ["-- None --"] + non_toxic_samples)

user_text = ""
if toxic_choice != "-- None --":
    user_text = toxic_choice
elif non_toxic_choice != "-- None --":
    user_text = non_toxic_choice

user_text = st.text_area("Enter a comment", user_text, height=120)

if st.button("Predict"):
    if user_text.strip() == "":
        st.warning("Please enter or select a comment.")
    else:
        cleaned = clean_text(user_text)
        out = classifier(cleaned)[0]

        label = LABEL_MAP[out["label"]]
        score = out["score"]

        st.subheader("Prediction")
        st.markdown(f"### **{label}**")
        st.write(f"Confidence: **{score:.3f}**")
        st.progress(score)

        with st.expander("View Raw Output"):
            st.json(out)

st.markdown("---")
st.subheader("📊 Model Evaluation")

metrics = {
    "Metric": ["Loss", "Accuracy", "Precision", "Recall", "F1 Score"],
    "Value": [0.1062, 0.9685, 0.8337, 0.8292, 0.8314],
}

df = pd.DataFrame(metrics)
st.table(df)

st.caption("Model trained for 2 epochs on DistilBERT.")
