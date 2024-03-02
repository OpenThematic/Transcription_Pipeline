import json
from matplotlib import colors
import numpy as np

def load_json_data(filepath):
    """Load JSON data from a file."""
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def map_probability_to_color(probability, color_map):
    """Map a probability to a color using a colormap."""
    return colors.rgb2hex(color_map(probability))

def calculate_word_level_confidence(segment):
    """Calculate the word-level confidence (average probability) for a segment."""
    if not segment["words"]:
        return 0
    return np.mean([word["probability"] for word in segment["words"]])

def generate_html_content(data, normalization_mode='global'):
    """Generate HTML content with styled words and segments, including word-level confidence scores."""
    # Setup for visualization styles
    prob_color_map = colors.LinearSegmentedColormap.from_list("prob_colormap", ["red", "orange", "yellow", "green"])
    conf_color_map = colors.LinearSegmentedColormap.from_list("conf_colormap", ["red", "green"])
    background_color = "#333333"
    text_color = "#FFFFFF"
    font_family = "Arial, sans-serif"
    
    # Calculate and normalize segment-level and word-level confidence scores
    segment_confidences = [np.exp(segment["avg_logprob"]) for segment in data["segments"] if segment["text"]]
    word_confidences = [calculate_word_level_confidence(segment) for segment in data["segments"] if segment["text"]]
    
    min_seg_conf, max_seg_conf = min(segment_confidences), max(segment_confidences)
    min_word_conf, max_word_conf = min(word_confidences), max(word_confidences)
    
    normalize = lambda x, min_x, max_x: (x - min_x) / (max_x - min_x) if max_x > min_x else 0
    
    html_content = f'<html><body style="background-color: {background_color}; color: {text_color}; font-family: {font_family};">'
    for segment, seg_conf, word_conf in zip(data["segments"], segment_confidences, word_confidences):
        if not segment["text"]:  # Skip empty segments
            continue
        
        normalized_seg_conf = normalize(seg_conf, min_seg_conf, max_seg_conf)
        normalized_word_conf = normalize(word_conf, min_word_conf, max_word_conf)
        
        seg_conf_color = map_probability_to_color(normalized_seg_conf, conf_color_map)
        word_conf_color = map_probability_to_color(normalized_word_conf, conf_color_map)
        
        # Display both segment-level and word-level confidence scores
        html_content += f'<p>[{segment["start"]:.2f}] <span style="background-color:{seg_conf_color};">SegConf: [{seg_conf:.2f}]</span> '
        html_content += f'<span style="background-color:{word_conf_color};">WordConf: [{word_conf:.2f}]</span> '
        
        for word in segment["words"]:
            prob = normalize(word["probability"], min_word_conf, max_word_conf)
            color = map_probability_to_color(prob, prob_color_map)
            html_content += f'<span style="border-bottom: 3px solid {color};">{word["word"]}</span> '
        html_content += '</p>'
    html_content += '</body></html>'
    
    return html_content

def save_html_content(html_content, output_filepath):
    """Save the generated HTML content to a file."""
    with open(output_filepath, 'w') as file:
        file.write(html_content)

# Example usage
if __name__ == "__main__":
    filepath = 'Results/results-wlarge-v3-beamtempstep2-translate/Kerstin, Lisbeth, Margareta - Björke Socken/Kerstin, Lisbeth, Margareta - Björke Socken_transcription_raw.json'
    output_filepath = filepath.replace('.json', '.html')
    normalization_mode = 'global'  # Choose 'global' or 'local' for probability normalization
    data = load_json_data(filepath)
    html_content = generate_html_content(data, normalization_mode)
    save_html_content(html_content, output_filepath)
    print("HTML content with confidence variation generated and saved.")
    




