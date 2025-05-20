import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pickle

@st.cache_resource
def load_model_and_tokenizer(model_folder="./model"):
    tokenizer = AutoTokenizer.from_pretrained(model_folder)
    model = AutoModelForSequenceClassification.from_pretrained(model_folder)
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_label_encoder(path="label_encoder.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer, model = load_model_and_tokenizer()
model = model.to(device)
label_encoder = load_label_encoder()

st.title("Dashboard Klasifikasi Sentimen")

user_input = st.text_area("Masukkan kalimat:")

if st.button("Prediksi"):
    if user_input.strip() == "":
        st.warning("Tolong masukkan teks sebelum klik Prediksi.")
    else:
        try:
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = np.argmax(probs)
                pred_label = label_encoder.inverse_transform([pred_idx])[0]

            st.markdown(f"### Prediksi Label: `{pred_label}`")
            st.markdown("#### Probabilitas per label:")
            for label, prob in zip(label_encoder.classes_, probs):
                st.write(f"- **{label}** : {prob * 100:.2f}%")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi: {e}")
