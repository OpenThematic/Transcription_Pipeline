#Addl modules
import audio_processing_in_memory
import transcription_stablets
import speaker_diarization
import utilities

import logging
import configparser
import os
import json
import glob
import sys


# Load configurations
config = configparser.ConfigParser()
config.read('config.ini')

# Set up logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def construct_output_paths(input_file):
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    results_dir = os.path.join(config.get('Results', 'ResultsDir'), base_name)
    return {
        'audio': os.path.join(results_dir, f"{base_name}_processed_audio.wav"),
        'diarization': os.path.join(results_dir, f"{base_name}_diarization.json"),
        'transcription': os.path.join(results_dir, f"{base_name}_transcription.json"),
        'transcription_raw': os.path.join(results_dir, f"{base_name}_transcription_raw.json"),
        'mapped_T_D': os.path.join(results_dir, f"{base_name}_mapped_T_D.json"),
        'final': os.path.join(results_dir, f"{base_name}_final_results.json"),
        'srt': os.path.join(results_dir, f"{base_name}_final_results.srt"),  
        'text': os.path.join(results_dir, f"{base_name}_final_results.txt"),
        'mp4' : os.path.join(results_dir, f"{base_name}_final_results.mp4")  
    }

def transcribe_all_files(input_dir):
    #Iterate over all files in the input dir, load the audio file into memory via audio_processing_in_memory.load_resample_trim_audio(input_file) and call transcription_stablets.transcribe_audio(audio in memory) then save the result of transcribe_audio to disk:

    logging.info(f"Transcribing audio in directory {input_dir}")

    for input_file in glob.glob(os.path.join(input_dir, '*')):
        logging.info(f"Processing audio in {input_file}")

        paths = construct_output_paths(input_file)

        # Ensure 'paths' is defined outside of any conditional blocks
        # Proceed with ensuring the directory exists
        os.makedirs(os.path.dirname(paths['audio']), exist_ok=True)

        logging.info(f"Preprocessing and loading {input_file}")
        # Audio processing via audio_processing_in_memory.load_resample_trim_audio(input_file)
        try:
            audio_loaded_to_memory = audio_processing_in_memory.load_resample_trim_audio(input_file)
        except Exception as e:
            logging.error(f"Main: Error processing audio {input_file}: {e}")
            continue

        logging.info(f"Transcribing {input_file}")
        # Transcription via transcription_stablets.transcribe_audio(audio in memory)
        try:
            transcription_results = transcription_stablets.transcribe_audio(audio_loaded_to_memory)
            transcription_results.save_as_json(paths['transcription_raw'])
        except Exception as e:
            logging.error(f"Main: Error transcribing audio {input_file}: {e}")
            continue

def diarize_audio(input_dir):
    #if not file_exists(paths['diarization']):
    for input_file in glob.glob(os.path.join(input_dir, '*')):
        paths = construct_output_paths(input_file)
        logging.info(f"Diarizing audio in directory {input_dir}")
        try:
            # Use None as a fallback if they are not specified or not integers
            num_speakers = config.getint('Diarization', 'NumSpeakers', fallback=None)
            min_speakers = config.getint('Diarization', 'MinSpeakers', fallback=None)
            max_speakers = config.getint('Diarization', 'MaxSpeakers', fallback=None) 
            # Try to diarize based on the range of possible speakers
            diarization_results = speaker_diarization.diarize_audio(
                input_file, 
                num_speakers = num_speakers, 
                min_speakers = min_speakers, 
                max_speakers = max_speakers
                )
            #Save to file
            #diarization_results.save_as_json(paths['diarization'])
            utilities.save_results_to_file(diarization_results, paths['diarization'])
        except Exception as e:
            logging.error(f"Error during speaker diarization: {e}")

def map_speakers_to_transcription(input_dir):
    for input_file in glob.glob(os.path.join(input_dir, '*')):

        paths = construct_output_paths(input_file)
        transcription_results = utilities.load_results_from_file(paths['transcription_raw'])
        diarization_results = utilities.load_results_from_file(paths['diarization'])
        overlap_threshold = config.getfloat('Diarization', 'OverlapThreshold', fallback=0.5)
        #print(f"diarization results: {diarization_results}")
    
        #if os.path.exists(paths['transcription']) and os.path.exists(paths['diarization']):


            #  For the transcription_results, iterate through each transcription_results.segments "start" and "end" and load them into variables

        for segment in transcription_results["segments"]:
            # Find the corresponding mapped_result for this segment
            transcript_start = segment['start']
            transcript_end = segment['end']
            print(f"Transcript segment start: {transcript_start}, transcript segment end: {transcript_end}")
            for diarization_segment in diarization_results:
                print(f"Diarization segment start: {diarization_segment['speaker_start']}, diarization segment end: {diarization_segment['speaker_end']}")
                # Calculate the overlap duration and percentage
                overlap_duration = max(0, min(segment['end'], diarization_segment['speaker_end']) - max(segment['start'], diarization_segment["speaker_start"]))
                print(f"Overlap duration: {overlap_duration}")
                segment_duration = segment['end'] - segment['start']
                print(f"Segment duration: {segment_duration}")
                overlap_percentage = overlap_duration / segment_duration
                print(f"Overlap percentage: {overlap_percentage}")
                # Check if the overlap percentage meets the threshold
                if overlap_percentage >= overlap_threshold:
                    # Add the speaker name to the segment dictionary
                    segment['speaker'] = diarization_segment['speaker']
                    # Break the inner loop once the match is found
                    break
        utilities.save_results_to_file(transcription_results, paths['mapped_T_D'])


    return None
 



def main(input_dir):
    # Iterate over audio files in the specified input directory
    logging.info(f"Processing audio in directory {input_dir}")

    if config.getboolean('General', 'Transcribe'):
        results = transcribe_all_files(input_dir)
    if config.getboolean('General', 'Diarize'):
        diarize_audio(input_dir)
    if config.getboolean('General', 'MapSpeakers'):
        map_speakers_to_transcription(input_dir)
    #if config.getboolean('General', 'LLM'):

            

# Example usage
if __name__ == "__main__":
    try:
        input_dir = config.get('General', 'InputDir', fallback='Input_AV')  # Provide a default path in case it's not specified
        main(input_dir)
    except Exception as e:
        logging.error(f"Failed to start processing: {e}")
