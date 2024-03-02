# path/filename: transcribe_swedish.py
import sys
import torch
import torchaudio
from datasets import load_dataset
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

def transcribe_swedish(audio_file_path):
    """
    Transcribe Swedish audio file to text using Wav2Vec 2.0 model.
    
    Args:
        audio_file_path (str): Path to the audio file to transcribe.
        
    Returns:
        str: The transcription of the audio file.
    """
    # Initialize processor and model
    processor = Wav2Vec2Processor.from_pretrained("KBLab/wav2vec2-large-voxrex-swedish")
    model = Wav2Vec2ForCTC.from_pretrained("KBLab/wav2vec2-large-voxrex-swedish")
    resampler = torchaudio.transforms.Resample(orig_freq=48_000, new_freq=16_000)
    
    # Load and preprocess the audio file
    speech_array, sampling_rate = torchaudio.load(audio_file_path)
    speech_array = resampler(speech_array).squeeze().numpy()
    
    # Prepare the audio file for the model
    inputs = processor(speech_array, sampling_rate=16_000, return_tensors="pt", padding=True)
    
    # Perform transcription
    with torch.no_grad():
        logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    
    # Decode the predicted ids to text
    transcription = processor.batch_decode(predicted_ids)
    
    return transcription

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python transcribe_swedish.py <audio_file_path>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    transcription = transcribe_swedish(audio_file)
    print("Transcription:", transcription)