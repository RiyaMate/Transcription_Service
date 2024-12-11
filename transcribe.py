from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile
from pathlib import Path
import whisper
import boto3
import os
from pyannote.audio import Pipeline
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fetch sensitive data from environment variables
HF_TOKEN = os.getenv("HF_TOKEN", "default_hf_token")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY", "default_access_key")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY", "default_secret_key")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME", "default_bucket_name")
AWS_REGION = os.getenv("AWS_REGION", "default_region")

# Set up FFmpeg path (if needed for your environment)
FFMPEG_PATH = r"C:\ffmeg\ffmpeg-7.0.2-essentials_build\bin\ffmpeg.exe"
os.environ["PATH"] += os.pathsep + os.path.dirname(FFMPEG_PATH)

# Initialize S3 client using credentials from environment variables
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

# Initialize FastAPI
app = FastAPI()

# Load Whisper model for transcription
whisper_model = whisper.load_model("base")

# Initialize Pyannote pipeline for diarization
try:
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN
    )
    print("Diarization pipeline loaded successfully.")
except Exception as e:
    diarization_pipeline = None
    print(f"Failed to load diarization pipeline: {e}")

# Function to upload files to S3
def upload_file_to_s3(file_path, s3_key):
    try:
        s3_client.upload_file(file_path, AWS_BUCKET_NAME, s3_key)
        s3_url = f"https://{AWS_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
        return s3_url
    except Exception as e:
        raise Exception(f"Failed to upload file to S3: {e}")

@app.post("/upload-audio/")
async def process_audio(file: UploadFile = File(...)):
    if diarization_pipeline is None:
        return JSONResponse(
            {"error": "Diarization pipeline unavailable. Please check your Hugging Face token."},
            status_code=500,
        )

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_audio:
            audio_data = await file.read()
            temp_audio.write(audio_data)
            audio_path = temp_audio.name

        # Upload the original audio file to S3
        s3_audio_key = f"audio_files/{file.filename}"
        s3_audio_url = upload_file_to_s3(audio_path, s3_audio_key)

        # Transcribe audio using Whisper
        transcription = whisper_model.transcribe(audio_path)["segments"]

        # Perform diarization using Pyannote pipeline
        diarization = diarization_pipeline(audio_path)

        # Combine transcription and diarization
        diarized_output = []
        for segment in transcription:
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"]
            speaker = "Unknown"
            for turn, _, speaker_id in diarization.itertracks(yield_label=True):
                if turn.start <= start_time <= turn.end:
                    speaker = f"Speaker {speaker_id}"
            diarized_output.append(f"[{start_time:.2f} - {end_time:.2f}] {speaker}: {text}")

        # Save diarized transcription to a file
        diarized_text = "\n".join(diarized_output)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_txt:
            temp_txt.write(diarized_text.encode("utf-8"))
            transcription_path = temp_txt.name

        # Upload diarized transcription to S3
        s3_transcription_key = f"transcriptions/{Path(file.filename).stem}_diarized.txt"
        s3_transcription_url = upload_file_to_s3(transcription_path, s3_transcription_key)

        # Clean up temporary files
        os.remove(audio_path)
        os.remove(transcription_path)

        return {
            "audio_file_s3_url": s3_audio_url,
            "transcription_file_s3_url": s3_transcription_url
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
