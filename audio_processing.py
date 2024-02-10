from pydub import AudioSegment
from pydub.effects import normalize
import yt_dlp
import logging
import configparser
import numpy as np
import noisereduce as nr
import subprocess

# Load configurations
config = configparser.ConfigParser()
config.read('config.ini')

def extract_audio_from_video(input_video_file, output_audio_file):
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': config.get('Audio', 'Codec'),
                'preferredquality': config.get('Audio', 'Quality'),
            }],
            'outtmpl': output_audio_file,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([input_video_file])
        return True
    except Exception as e:
        logging.error(f"Error in extracting audio: {e}")
        return False

def trim_audio(input_audio_file, output_audio_file, duration_minutes):
    try:
        # Load audio
        audio = AudioSegment.from_file(input_audio_file)
        trimmed_audio = audio[:duration_minutes * 60 * 1000]

        # Noise reduction
        if config.getboolean('Audio', 'EnableNoiseReduction'):
            # Convert AudioSegment to numpy array
            samples = np.array(trimmed_audio.get_array_of_samples())
            
            # Reduce noise
            reduced_noise = nr.reduce_noise(samples, sr=trimmed_audio.frame_rate, prop_decrease=0.3, n_std_thresh_stationary=2.25, n_fft=512, win_length=512)

            # Convert back to AudioSegment
            reduced_noise_audio = AudioSegment(
                data=reduced_noise.tobytes(),
                sample_width=trimmed_audio.sample_width,
                frame_rate=trimmed_audio.frame_rate,
                channels=trimmed_audio.channels
            )
            trimmed_audio = reduced_noise_audio

        # Volume normalization
        if config.getboolean('Audio', 'EnableVolumeNormalization'):
            trimmed_audio = normalize(trimmed_audio)

             #Export the processed audio
        trimmed_audio.export(output_audio_file, format="wav")
       
        return True
    except Exception as e:
        logging.error(f"Error in processing audio: {e}")
        return False

def combine_audio_subtitles(audio_file, srt_file, output_file):
    try:
        subprocess.call([
            'ffmpeg', 
            '-i', audio_file, 
            '-i', srt_file, 
            '-c:a', 'aac', 
            '-b:a', '192k', 
            '-vn', 
            '-c:s', 'mov_text', 
            output_file
        ])
        logging.info(f"Successfully created {output_file} with audio and subtitles.")
    except Exception as e:
        logging.error(f"Failed to create {output_file} with audio and subtitles: {e}")
