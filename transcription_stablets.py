import logging
import configparser
import stable_whisper

#https://github.com/jianfch/stable-ts?tab=readme-ov-file#transcribe

# Load configurations
config = configparser.ConfigParser()
config.read('config.ini')

# Load the Whisper model based on configuration
modelId = config.get('Whisper', 'ModelID')
model =  stable_whisper.load_model(modelId)


def transcribe_audio(input_audio):
    try:
        result = model.transcribe(input_audio, verbose=True, word_timestamps=True, vad=True)
    except Exception as e:
        logging.error(f"StableTS Error in transcribing audio: {e}")
        return None
    return result




