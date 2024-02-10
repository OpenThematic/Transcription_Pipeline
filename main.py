import audio_processing
import speaker_diarization
import transcription
import logging
import configparser
import os
import json
import glob

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configurations
config = configparser.ConfigParser()
config.read('config.ini')

class CriticalError(Exception):
    pass

def save_results_to_file(results, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure the directory exists
    try:
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=4)
    except IOError as e:
        logging.error(f"Failed to save results to {file_path}: {e}")

def load_results_from_file(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except IOError as e:
        logging.error(f"Failed to load results from {file_path}: {e}")
        return None

def file_exists(file_path):
    return os.path.exists(file_path)

def map_speakers_to_transcription(transcription_results, diarization_results, overlap_threshold):
    mapped_results = []

    for transcript in transcription_results:
        transcript_start = transcript["start"]
        transcript_end = transcript["end"]
        transcript_duration = transcript_end - transcript_start

        # Find overlapping diarization segments and calculate overlap duration
        overlaps = []
        for speaker in diarization_results:
            overlap_start = max(speaker["start"], transcript_start)
            overlap_end = min(speaker["end"], transcript_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            if overlap_duration > 0:
                overlaps.append((speaker["speaker"], overlap_duration))

        # Calculate total overlap duration to find the majority speaker
        total_overlap = sum(duration for _, duration in overlaps)

        # Determine the speaker based on the majority overlap
        majority_speaker = None
        majority_overlap = 0
        for speaker, duration in overlaps:
            if duration / total_overlap > majority_overlap:
                majority_speaker = speaker
                majority_overlap = duration / total_overlap

        # Check if the majority overlap meets the threshold
        if majority_overlap < overlap_threshold or majority_speaker is None:
            speaker_id = "OVERLAPPED"
        else:
            speaker_id = majority_speaker

        mapped_results.append({
            "start": transcript_start,
            "end": transcript_end,
            "speaker": speaker_id,
            "text": transcript["text"]
        })

    return mapped_results

def save_results_to_srt(results, srt_file_path):
    os.makedirs(os.path.dirname(srt_file_path), exist_ok=True)  # Ensure the directory exists
    try:
        with open(srt_file_path, 'w') as f:
            for i, result in enumerate(results, start=1):
                start = format_time(result['start'])
                end = format_time(result['end'])
                speaker_number = result['speaker'].replace('SPEAKER_', '')  # Remove 'SPEAKER_' prefix
                text = f"({speaker_number}) {result['text']}"  # Include speaker number in parentheses
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
    except IOError as e:
        logging.error(f"Failed to save SRT to {srt_file_path}: {e}")

def save_results_to_text(results, text_file_path):
    os.makedirs(os.path.dirname(text_file_path), exist_ok=True)  # Ensure the directory exists
    try:
        with open(text_file_path, 'w') as f:
            for result in results:
                # Format start time as [HH:MM:SS]
                start_time = format_time_simple(result['start'])
                # Include speaker label and text, minimize timestamp usage
                f.write(f"[{start_time}] {result['speaker']}: {result['text']}\n")
    except IOError as e:
        logging.error(f"Failed to save human-readable text to {text_file_path}: {e}")


def format_time(seconds):
    """Convert seconds to SRT time format."""
    millisec = int((seconds - int(seconds)) * 1000)
    return f"{int(seconds // 3600):02}:{int(seconds % 3600 // 60):02}:{int(seconds % 60):02},{millisec:03}"

def format_time_simple(seconds):
    """Convert seconds to a simpler HH:MM:SS time format."""
    return f"{int(seconds // 3600):02}:{int(seconds % 3600 // 60):02}:{int(seconds % 60):02}"


def construct_output_paths(input_file):
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    results_dir = os.path.join(config.get('Results', 'ResultsDir'), base_name)
    return {
        'audio': os.path.join(results_dir, f"{base_name}_processed_audio.wav"),
        'diarization': os.path.join(results_dir, f"{base_name}_diarization.json"),
        'transcription': os.path.join(results_dir, f"{base_name}_transcription.json"),
        'transcription_raw': os.path.join(results_dir, f"{base_name}_transcription_raw.json"),
        'final': os.path.join(results_dir, f"{base_name}_final_results.json"),
        'srt': os.path.join(results_dir, f"{base_name}_final_results.srt"),  
        'text': os.path.join(results_dir, f"{base_name}_final_results.txt"),
        'mp4' : os.path.join(results_dir, f"{base_name}_final_results.mp4")  
    }

import sys  # Make sure this import is at the top of your script

def main(input_dir):
    # Iterate over audio files in the specified input directory
    logging.info(f"Processing audio in directory {input_dir}")

    for input_file in glob.glob(os.path.join(input_dir, '*')):
        logging.info(f"Processing audio in {input_file}")

        # Ensure 'paths' is defined outside of any conditional blocks
        paths = construct_output_paths(input_file)

        # Proceed with ensuring the directory exists
        os.makedirs(os.path.dirname(paths['audio']), exist_ok=True)

        try:
            logging.info(f"About to try processing the file")
            # Audio processing and transcription logic
            if input_file.endswith(('.mp4', '.mkv', '.avi')):
                audio_processing.extract_audio_from_video(input_file, paths['audio'])
            audio_processing.trim_audio(input_file, paths['audio'], config.getint('General', 'DurationMinutes'))


            print("Step 2: Transcription")
            if not file_exists(paths['transcription']):
                try:
                    transcription_results = transcription.transcribe_audio(paths['audio'])
                    save_results_to_file(transcription_results, paths['transcription_raw'])
                    
                    # Extract and format transcription results
                    transcription_segments = []
                    for segment in transcription_results["segments"]:
                        transcription_segments.append({
                            "start": segment["start"],
                            "end": segment["end"],
                            "text": segment["text"]
                        })
                    transcription_results = transcription_segments
                    save_results_to_file(transcription_segments, paths['transcription'])
                except Exception as e:
                    logging.error(f"Error during transcription: {e}")                 

            print("Step 3: Speaker Diarization")
            if not file_exists(paths['diarization']):
                try:
                    # Use None as a fallback if they are not specified or not integers
                    num_speakers = config.getint('Diarization', 'NumSpeakers', fallback=None)
                    min_speakers = config.getint('Diarization', 'MinSpeakers', fallback=None)
                    max_speakers = config.getint('Diarization', 'MaxSpeakers', fallback=None) 

                    diarization_results = speaker_diarization.diarize_audio(
                        paths['audio'], 
                        num_speakers=num_speakers, 
                        min_speakers=min_speakers, 
                        max_speakers=max_speakers
                        )
                    save_results_to_file(diarization_results, paths['diarization'])
                except Exception as e:
                    logging.error(f"Error during speaker diarization: {e}")

        except Exception as e:
            logging.error(f"Error during processing: {e}")


        print("Step 4: Matching Diarization with Transcription")
        try:
            if diarization_results and transcription_results:  # Ensure both results are available
                overlap_threshold = float(config.get('Diarization', 'OverlapThreshold', fallback='0.5'))  # Default to 0.5 if not specified
                #final_results = match_diarization_with_transcription(diarization_results, transcription_results)
                final_results = map_speakers_to_transcription(transcription_results, diarization_results, overlap_threshold)
                save_results_to_file(final_results, paths['final']) # Save final results
                save_results_to_srt(final_results, paths['srt'])  # Save results to SRT
                save_results_to_text(final_results, paths['text']) # Save results to human readable text
                audio_processing.combine_audio_subtitles(paths['audio'], paths['srt'], paths['mp4']) #combine into MP4

                #Cleanup
                if os.path.exists(paths['audio']):
                    os.remove(paths['audio'])
                    logging.info(f"Temporary audio file {paths['audio']} removed successfully.")

            else:
                logging.error("Diarization or transcription results are missing, cannot proceed to matching.")

        # Move processed file to results directory
        #os.rename(input_file, os.path.join(paths['results'], os.path.basename(input_file)))
        #logging.info(f"Processed file moved to {paths['results']}")

        except Exception as e:
            logging.error(f"Error during matching diarization with transcription: {e}")


# Example usage
if __name__ == "__main__":
    try:
        input_dir = config.get('General', 'InputDir', fallback='Input_AV')  # Provide a default path in case it's not specified
        main(input_dir)
    except Exception as e:
        logging.error(f"Failed to start processing: {e}")