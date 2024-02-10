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

## Installation



Usage
Place your audio or video files in a directory accessible by the script.
Run the main.py script with the path to your file:
bash
Copy code
python main.py path_to_your_file.mp4
Arguments:

input_file: Path to the input audio or video file.
is_video (optional): Set to True if the input file is a video (default: True).
duration_minutes (optional): Duration in minutes to trim the audio (default: 20).
Modules
audio_processing.py: Handles audio extraction and trimming.
speaker_diarization.py: Performs speaker diarization using pyannote.audio.
transcription.py: Transcribes audio using Whisper.
main.py: Integrates all modules and runs the pipeline.
Contributing
Contributions to improve the project are welcome. Please follow the standard fork and pull request workflow.

License
[Your chosen license]

Acknowledgments
OpenAI for providing the Whisper model.
Contributors and maintainers of pyannote.audio, pydub, and yt_dlp.