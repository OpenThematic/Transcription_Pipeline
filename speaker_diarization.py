import json
import logging
import torch
from pyannote.audio import Pipeline
import configparser
import torchaudio 

# Load configurations
config = configparser.ConfigParser()
config.read('config.ini')

def diarize_audio(input_audio_file, num_speakers=None, min_speakers=None, max_speakers=None):
    try:
        # Load the pre-trained diarization pipeline with configurations
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=config.get('Models', 'AuthToken')
        )

        # Ensure GPU if possible
        pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # Load audio into memory for faster processing (optional)
        waveform, sample_rate = torchaudio.load(input_audio_file)
        audio_data = {"waveform": waveform, "sample_rate": sample_rate}

        # Run the pipeline with progress monitoring
        from pyannote.audio.pipelines.utils.hook import ProgressHook
        with ProgressHook() as hook:
            diarization_options = {"hook": hook}
            if num_speakers:
                diarization_options["num_speakers"] = num_speakers
            if min_speakers:
                diarization_options["min_speakers"] = min_speakers
            if max_speakers:
                diarization_options["max_speakers"] = max_speakers

            diarization = pipeline(audio_data, **diarization_options)
        

        diarization_results = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            diarization_results.append({
                "speaker_start": turn.start,
                "speaker_end": turn.end,
                "speaker": speaker
            })

        
        return diarization_results
    except Exception as e:
        logging.error(f"Error in speaker diarization: {e}")
        return []