import streamlit as st
import tempfile
import os
import torch

# Import model and functions from predict.py
from src.predict import extract_features, model, device, CLASS_NAMES

st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️", layout="centered")

st.title("🎙️ Speech Emotion Recognition")
st.write("Upload or drag and drop an audio file, and the model will predict the emotion!")

emojis = {
    "neutral": "😐",
    "calm": "😌",
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "fearful": "😨",
    "disgust": "🤢",
    "surprised": "😲"
}

uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'ogg', 'flac'])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    if st.button("Predict Emotion", type="primary", use_container_width=True):
        with st.spinner("Analyzing audio..."):
            # Save uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                features = extract_features(tmp_path)
                inputs = torch.from_numpy(features).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    logits = model(inputs)
                    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                    pred_idx = probs.argmax()
                    
                emotion = CLASS_NAMES[pred_idx]
                confidence = probs[pred_idx] * 100
                emoji = emojis.get(emotion, "🤖")
                
                st.markdown("---")
                st.markdown(f"<h1 style='text-align: center; font-size: 120px; margin-bottom: -20px;'>{emoji}</h1>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='text-align: center; margin-top: 0; color: #1f77b4;'>{emotion.upper()}</h2>", unsafe_allow_html=True)
                st.markdown(f"<p style='text-align: center; font-size: 20px; color: #7f8c8d;'>Confidence: <b>{confidence:.2f}%</b></p>", unsafe_allow_html=True)
                st.markdown("---")
                
            except Exception as e:
                st.error(f"Error processing audio: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
