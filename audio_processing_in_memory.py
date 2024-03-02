import soundfile as sf
import torchaudio
import numpy as np
import configparser
import logging

from pydub import AudioSegment
import io

def load_resample_trim_audio(input_audio_file):
    # load the config
    logging.info(f"Audio-Pre-Process: Loading config")
    config = configparser.ConfigParser()
    config.read('config.ini')
    audio_preprocess = config['Audio_Preprocess']
    start_sec = float(audio_preprocess.get('start_sec', None))
    end_sec = float(audio_preprocess.get('end_sec', None))
    sample_rate = int(audio_preprocess.get('SampleRate', 16000))
    
    
    # Load the audio file
    logging.info(f"Audio-Pre-Process: Loading audio file {input_audio_file}")
    try:
        loaded_audio = AudioSegment.from_file(input_audio_file)
        loaded_audio_trimmed = loaded_audio[start_sec * 1000 :  end_sec * 1000]
        loaded_audio_converted = loaded_audio_trimmed.set_frame_rate(sample_rate)
        # Create an in-memory buffer
        buffer = io.BytesIO()
        # Export the audio as wav to the buffer
        loaded_audio_converted.export(buffer, format="wav")
        # Get the buffer content as bytes
        wav_bytes = buffer.getvalue()

    except Exception as e:
        print(f"PreProcess Error in loading audio file: {e}")
        return None
    
    return wav_bytes

    
    # Convert start and end times from seconds to samples
    #if start_sec is not None and end_sec is not None:
    #    try:
    #        print(f"Audio-Pre-Process: Trimming audio to {start_sec} - {end_sec} seconds")
    #        start_sample = int(start_sec * sample_rate)
    #        end_sample = min(int(end_sec * sample_rate), len(audio))  # Ensure end_sample does not exceed audio length
    #    except Exception as e:
    #        print(f"Error in converting start and end times from seconds to samples: {e}")
    #        return None
    

    


# Example usage
#input_audio_file = 'path/to/your/audio.ogg'
#start_sec, end_sec, sample_rate = load_audio_config()
#audio_for_stable_ts = load_resample_trim_audio(input_audio_file, start_sec, end_sec, sample_rate)

# Now, audio_for_stable_ts is ready to be used with stable-ts and contains only the desired segment
