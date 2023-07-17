import boto3
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
import tempfile
import os

s3_bucket = "your-s3-bucket"  # Replace with your S3 bucket name
data_cache = "/path/to/data_cache"  # Replace with the path to your data cache directory

os.makedirs(data_cache, exist_ok=True)  # Create the data cache directory if it doesn't exist

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts", cachedir=data_cache)
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts", cachedir=data_cache)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan", cachedir=data_cache)

def lambda_handler(event, context):
    text = event["text"]
    
    inputs = processor(text=text, return_tensors="pt")

    # Load xvector containing speaker's voice characteristics from a dataset
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation", cache_dir=data_cache)
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

    # Save the speech to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
        temp_file_path = temp_file.name
        sf.write(temp_file_path, speech.numpy(), samplerate=16000)
        
        # Upload the file to S3
        s3_key = "speech.wav"  # Replace with your desired S3 key
        s3_client = boto3.client("s3")
        s3_client.upload_file(temp_file_path, s3_bucket, s3_key)
    
    return {
        "statusCode": 200,
        "body": {
            "message": "Speech file uploaded to S3",
            "s3_bucket": s3_bucket,
            "s3_key": s3_key
        }
    }
