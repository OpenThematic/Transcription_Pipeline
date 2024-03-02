# path/filename: modified_code_with_config_parser.py
import configparser
import torch
from transformers import pipeline, WhisperProcessor
import pprint
import soundfile as sf

# Load configurations
config = configparser.ConfigParser()
config.read('config.ini')

def transcribe_audio(input_audio_file):
    # Load device and model configurations
    try:
        hardware_config = config.get('CPU_GPU', 'Hardware', fallback='CPU').upper()
        device = "cuda" if hardware_config == 'GPU' and torch.cuda.is_available() else "cpu"
        model_id = config.get('Whisper', 'ModelID', fallback='openai/whisper-large-v3')
    except Exception as e:
        print(f"Error in loading device and model configurations: {e}")
        return None

    # Initialize the Whisper model and processor
    try:
        processor = WhisperProcessor.from_pretrained(model_id)
    except Exception as e:
        print(f"Error in initializing the Whisper Processor: {e}")
        return None

    print("setting pipe")
    try:
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            #chunk_length_s=30,
            #batch_size=16,
            return_timestamps=True, #
            device=device,
        )
    except Exception as e:
        print(f"Error in initializing the Whisper Pipeline: {e}")
        return None

    # Load the input audio file and transcribe
    try:    
        print("loading audio")
        audio_array, sampling_rate = sf.read(input_audio_file)
    except Exception as e:
        print(f"Error in loading audio: {e}")
        return None
    try:
        print("transcribing")
        result = pipe(audio_array)
    except Exception as e:
        print(f"Error in transcribing: {e}")
        return None
    try:
        print("done")
        pprint.pprint(result["text"])
    except Exception as e:
        print(f"Error in printing: {e}")
        return None

    return result["text"]
    
