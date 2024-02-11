# Speaker Identification and Transcription Pipeline

This project provides a Python-based pipeline for extracting audio from video files, performing speaker diarization, and transcribing audio using OpenAI's Whisper. It identifies different speakers in the audio and aligns transcriptions with these speakers.

## Features

- Audio extraction from video files
- Audio trimming to a specified duration
- Speaker diarization to identify different speakers in the audio
- Audio transcription using OpenAI's Whisper
- Matching transcriptions with speaker diarization data

## Prerequisites

- Python 3.x
- Libraries: `pydub`, `yt_dlp`, `pyannote.audio`, `whisper`
- ffmpeg (for audio extraction from video)
- Ideally CUDA GPU with  10GB for whisper large or 6-8gb for whisper medium - will run on CPU, just slower.
- See docker file

## Installation
Ideally docker compose a dev environment via the compose yaml / dockerfile


Usage
Configuration is made in config.ini - there you can set a path to your input directory (where audio or video files are stored).
Make changes (other than the model) to the transcription settings within transcript.py.
Run the main.py script.

audio_processing.py: Handles audio extraction and trimming, noise reduction, normalization, produces a mp4 with embedded subtitles.
speaker_diarization.py: Performs speaker diarization using pyannote.audio.
transcription.py: Transcribes audio using Whisper.
main.py: Integrates all modules and runs the pipeline - produces various output formats.


License
[Your chosen license]

Acknowledgments
OpenAI for providing the Whisper model.
Contributors and maintainers of pyannote.audio, pydub, and yt_dlp.
