import os
import json
import logging
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

def format_time(seconds):
    """Convert seconds to SRT time format."""
    millisec = int((seconds - int(seconds)) * 1000)
    return f"{int(seconds // 3600):02}:{int(seconds % 3600 // 60):02}:{int(seconds % 60):02},{millisec:03}"

def format_time_simple(seconds):
    """Convert seconds to a simpler HH:MM:SS time format."""
    return f"{int(seconds // 3600):02}:{int(seconds % 3600 // 60):02}:{int(seconds % 60):02}"