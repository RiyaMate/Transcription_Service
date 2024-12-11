###FRONTEND to test

import streamlit as st
import requests
import os

# Backend API URL
BACKEND_URL = "http://127.0.0.1:8000/upload-audio/"

# Title and description
st.title("Audio Transcription and Diarization")
st.write("Upload an audio file to transcribe it with speaker diarization. The processed transcription will be uploaded to S3.")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a", "flac"])

if uploaded_file is not None:
    st.write(f"File uploaded: {uploaded_file.name}")

    # Save the uploaded file temporarily
    temp_file_path = os.path.join("temp_audio", uploaded_file.name)
    os.makedirs("temp_audio", exist_ok=True)
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())

    # Send file to backend
    with open(temp_file_path, "rb") as audio_file:
        files = {"file": audio_file}
        with st.spinner("Processing the audio file..."):
            response = requests.post(BACKEND_URL, files=files)

    # Clean up temporary files
    os.remove(temp_file_path)

    # Display response
    if response.status_code == 200:
        result = response.json()
        st.success("Transcription completed!")
        st.write("Diarized transcription file URL:")
        st.write(result["transcription_file_s3_url"])
    else:
        st.error("An error occurred while processing the audio file.")
        st.write(response.json()["error"])

