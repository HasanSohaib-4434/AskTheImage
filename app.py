import streamlit as st
import tempfile
import torch
import whisper
import pyttsx3
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoProcessor, AutoModelForVision2Seq
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import soundfile as sf

st.set_page_config(page_title="Ask the Image", layout="wide")
st.title("🧠 Ask-the-Image: Multimodal QA with Voice")

@st.cache_resource
def load_models():
    asr_model = whisper.load_model("small")
    vlm_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-small")
    vlm_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-small", device_map="auto", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    return asr_model, vlm_processor, vlm_model

asr_model, vlm_processor, vlm_model = load_models()

def transcribe_audio(audio_path):
    result = asr_model.transcribe(audio_path)
    return result["text"]

def generate_answer(image, question):
    inputs = vlm_processor(images=image, text=question, return_tensors="pt").to(vlm_model.device, torch.float16 if torch.cuda.is_available() else torch.float32)
    out = vlm_model.generate(**inputs, max_new_tokens=50)
    return vlm_processor.decode(out[0], skip_special_tokens=True)

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

st.sidebar.subheader("🎤 Record Your Question")
audio_file = st.sidebar.file_uploader("Upload 10s voice clip (WAV/MP3)", type=["wav", "mp3"])

if audio_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        tmp_audio.write(audio_file.read())
        tmp_audio_path = tmp_audio.name
    question_text = transcribe_audio(tmp_audio_path)
    st.sidebar.success(f"Transcribed: {question_text}")
else:
    question_text = st.sidebar.text_input("Or type your question")

uploaded_image = st.file_uploader("📷 Upload an Image", type=["jpg", "jpeg", "png"])
if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if question_text:
        st.markdown(f"**Question:** {question_text}")
        with st.spinner("Generating answer..."):
            answer = generate_answer(image, question_text)
        st.success(f"🗣️ Answer: {answer}")
        if st.button("🔊 Speak Answer"):
            speak_text(answer)
