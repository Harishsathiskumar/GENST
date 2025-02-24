import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import base64
from openai import OpenAI
from transformers import pipeline
import fitz  # PyMuPDF for ATS PDF parsing
import os

# Set up OpenAI API key (add yours in Streamlit secrets or environment variable)
openai_api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not openai_api_key:
    st.error("Please set your OpenAI API key in Streamlit secrets.")
    st.stop()
client = OpenAI(api_key=openai_api_key)

# Stable Diffusion API setup (using Hugging Face as an example - add your token)
hf_api_key = st.secrets.get("HF_API_KEY", os.getenv("HF_API_KEY"))
if not hf_api_key:
    st.warning("Hugging Face API key not set. Text-to-Image may not work.")

# App Title
st.title("AI-Powered Web Application")

# Tabs for each feature
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Text-to-Image",
    "Text-to-Audio",
    "Summarization",
    "Code Debugger",
    "ATS Score Checker"
])

# 1. Text-to-Image Generation
with tab1:
    st.header("Text-to-Image Generation")
    prompt = st.text_input("Enter a prompt to generate an image:", "A futuristic city at night")
    if st.button("Generate Image"):
        if hf_api_key:
            api_url = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
            headers = {"Authorization": f"Bearer {hf_api_key}"}
            payload = {"inputs": prompt}
            with st.spinner("Generating image..."):
                response = requests.post(api_url, headers=headers, json=payload)
                if response.status_code == 200:
                    image_bytes = response.content
                    image = Image.open(BytesIO(image_bytes))
                    st.image(image, caption="Generated Image")
                else:
                    st.error("Failed to generate image. Check API key or prompt.")
        else:
            st.error("Hugging Face API key required for Stable Diffusion.")

# 2. Text-to-Audio Conversion
with tab2:
    st.header("Text-to-Audio Conversion")
    text_input = st.text_area("Enter text to convert to audio:", "Hello, this is a test.")
    voice = st.selectbox("Choose voice:", ["alloy", "echo", "fable"])
    if st.button("Convert to Audio"):
        with st.spinner("Generating audio..."):
            response = client.audio.speech.create(model="tts-1", voice=voice, input=text_input)
            audio_file = "output.mp3"
            response.stream_to_file(audio_file)
            with open(audio_file, "rb") as f:
                audio_bytes = f.read()
            st.audio(audio_bytes, format="audio/mp3")
            st.download_button("Download Audio", audio_bytes, "output.mp3", "audio/mp3")

# 3. AI-Powered Summarization
with tab3:
    st.header("AI-Powered Summarization")
    long_text = st.text_area("Enter text to summarize:", "Paste your long text here...")
    if st.button("Summarize"):
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        with st.spinner("Summarizing..."):
            summary = summarizer(long_text, max_length=130, min_length=30, do_sample=False)
            st.write("Summary:", summary[0]["summary_text"])

# 4. AI-Based Code Debugger & Explainer
with tab4:
    st.header("AI-Based Code Debugger & Explainer")
    code_input = st.text_area("Paste your code here:", "def example():\n    print(undefined_variable)")
    if st.button("Debug & Explain"):
        with st.spinner("Analyzing code..."):
            debug_prompt = f"Analyze this code for errors and explain fixes:\n```python\n{code_input}\n```"
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": debug_prompt}]
            )
            explanation = response.choices[0].message.content
            st.markdown(explanation)

# 5. ATS Score Checker
with tab5:
    st.header("ATS Score Checker")
    resume_file = st.file_uploader("Upload your resume (PDF):", type="pdf")
    job_desc = st.text_area("Paste job description:", "Enter job description here...")
    if st.button("Check ATS Score"):
        if resume_file and job_desc:
            with st.spinner("Analyzing..."):
                # Extract text from PDF
                pdf = fitz.open(stream=resume_file.read(), filetype="pdf")
                resume_text = "".join(page.get_text() for page in pdf)
                # Simple keyword matching for demo (expand with NLP for production)
                resume_words = set(resume_text.lower().split())
                job_words = set(job_desc.lower().split())
                common = resume_words.intersection(job_words)
                score = min(len(common) / len(job_words) * 100, 100)
                st.write(f"ATS Score: {score:.2f}%")
                st.write("Matching keywords:", ", ".join(common))
        else:
            st.error("Please upload a resume and enter a job description.")

# Footer
st.markdown("---")
st.write("Built with Streamlit and powered by AI tools. Deployed on Streamlit Community Cloud.")
