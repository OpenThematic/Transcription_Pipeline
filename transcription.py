import whisper
import logging
import configparser

# Load configurations
config = configparser.ConfigParser()
config.read('config.ini')

# Load the Whisper model based on configuration
model_size = config.get('Whisper', 'ModelSize')
model = whisper.load_model(model_size)

"""
Will be converting to stable-whisper library for simplicity and consolidation of features.
"""

'''Simple transcription using OpenAI Whisper library
    Parameters
    ----------
    model: Whisper
        The Whisper model instance

    audio: Union[str, np.ndarray, torch.Tensor]
        The path to the audio file to open, or the audio waveform

    verbose: bool
        Whether to display the text being decoded to the console. If True, displays all the details,
        If False, displays minimal details. If None, does not display anything

    temperature: Union[float, Tuple[float, ...]]
        Temperature for sampling. It can be a tuple of temperatures, which will be successively used
        upon failures according to either `compression_ratio_threshold` or `logprob_threshold`.

    compression_ratio_threshold: float
        If the gzip compression ratio is above this value, treat as failed

    logprob_threshold: float
        If the average log probability over sampled tokens is below this value, treat as failed

    no_speech_threshold: float
        If the no_speech probability is higher than this value AND the average log probability
        over sampled tokens is below `logprob_threshold`, consider the segment as silent

    condition_on_previous_text: bool
        if True, the previous output of the model is provided as a prompt for the next window;
        disabling may make the text inconsistent across windows, but the model becomes less prone to
        getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.

    word_timestamps: bool
        Extract word-level timestamps using the cross-attention pattern and dynamic time warping,
        and include the timestamps for each word in each segment.

    prepend_punctuations: str
        If word_timestamps is True, merge these punctuation symbols with the next word

    append_punctuations: str
        If word_timestamps is True, merge these punctuation symbols with the previous word

    initial_prompt: Optional[str]
        Optional text to provide as a prompt for the first window. This can be used to provide, or
        "prompt-engineer" a context for transcription, e.g. custom vocabularies or proper nouns
        to make it more likely to predict those word correctly.

    decode_options: dict
        Keyword arguments to construct `DecodingOptions` instances

    clip_timestamps: Union[str, List[float]]
        Comma-separated list start,end,start,end,... timestamps (in seconds) of clips to process.
        The last end timestamp defaults to the end of the file.

    hallucination_silence_threshold: Optional[float]
        When word_timestamps is True, skip silent periods longer than this threshold (in seconds)
        when a possible hallucination is detected

    Returns
    -------
        A dictionary containing the resulting text ("text") and segment-level details ("segments"), and
    the spoken language ("language"), which is detected when `decode_options["language"]` is None.
    '''
def transcribe_audio(input_audio_file):

    options = {
    "verbose": True,
    
    # Specifies the language spoken in the audio. If set to None, Whisper will attempt to detect the language.
    # Default: None
    "language": "Swedish",

    # Determines whether to transcribe the audio as-is ("transcribe") or translate it into English ("translate").
    # Default: "transcribe"
    "task": "translate",

    # Sets the temperature for sampling; higher values generate more varied output. Zero disables sampling for deterministic output.
    # Default: 0.0
    "temperature": 0.2,

    # When using a non-zero temperature, this specifies the number of candidate transcriptions to consider.
    # Default: 5
    "best_of": 10,

    # Number of beams in beam search, applicable only when temperature is zero. More beams can improve accuracy at the cost of speed.
    # Default: 5
    "beam_size": 5,

    # Patience parameter for beam decoding, influencing how long to wait for a better option before finalizing a decision.
    # Default: None (equivalent to conventional beam search with a patience of 1.0)
    "patience": 2.0,

    # Applies a length penalty to discourage overly long or short sentences, affecting beam search.
    # Default: None (uses simple length normalization)
    "length_penalty": None,

    # A list of token IDs to suppress during sampling, preventing certain tokens from being generated.
    # Default: "-1" (suppresses most special characters except common punctuation)
    "suppress_tokens": "-1",

    # Initial prompt text to provide context or start the transcription, influencing the model's output.
    # Default: None
    "initial_prompt": "An interview between a researcher and residents of Gotland, Sweden.  The interview is conducted in Swedish and opens with introductions.",

    # Whether to use the previous output as a prompt for the next window, helping maintain context.
    # Default: True
    "condition_on_previous_text": True,

    # Whether to perform inference in FP16 mode, which can be faster on compatible hardware.
    # Default: True
    "fp16": True,

    # The amount to increase the temperature after a fallback due to failing the compression ratio or log probability thresholds.
    # Default: 0.2 
    #"temperature_increment_on_fallback": 0.2, #Was causing an error, debug - not a recognized argument
    

    # Threshold for the gzip compression ratio. Decodings with a higher ratio are considered failed and retried with different parameters.
    # Default: 2.4
    "compression_ratio_threshold": 2.4,

    # Threshold for the average log probability. Decodings below this threshold are considered low-confidence and may trigger a fallback.
    # Default: -1.0
    "logprob_threshold": -1.0,

    # Threshold for considering a segment as silence based on the probability of the no speech token, affecting how non-speech is handled.
    # Default: 0.6
    "no_speech_threshold": 0.6,

    # Extracts word-level timestamps, useful for detailed transcriptions or subtitles.
    # Default: False
    "word_timestamps": True,

    # Punctuation symbols to prepend to the next word, used with word timestamps.
    # Default: "\"\'“¿([{-"
    #"prepend_punctuations": "\"\'“¿([{-",

    # Punctuation symbols to append to the previous word, used with word timestamps.
    # Default: "\"\'.。,，!！?？:：”)]}、"
    #"append_punctuations": "\"\'.。,，!！?？:：”)]}、",

    # Underlines each word as it is spoken in subtitles, requires word timestamps.
    # Default: False
    #"highlight_words": False,  #was causing an error unexpected arg

    # Maximum number of characters in a line for subtitles, requires word timestamps.
    # Default: None
    #"max_line_width": None,

    # Maximum number of lines in a subtitle segment, requires word timestamps.
    # Default: None
    #"max_line_count": None,

    # Maximum number of words per line in subtitles, requires word timestamps and is not used with max_line_width.
    # Default: None
    #"max_words_per_line": None,

    # Number of threads used by PyTorch for CPU inference, can override environment variables like MKL_NUM_THREADS.
    # Default: 0 (PyTorch's default setting)
    #"threads": 0, #unexpected keyword

    # Specifies timestamps of clips to process in the format start,end,start,end,..., where the last end timestamp defaults to the end of the file.
    # Default: "0"
    #"clip_timestamps": "0",

    # Threshold for skipping silent periods longer than this value when a possible hallucination is detected, requires word timestamps.
    # Default: None
    #"hallucination_silence_threshold": 1
}


    try:
        # Transcribe the audio
        result = model.transcribe(input_audio_file, **options)
        return result
    except Exception as e:
        logging.error(f"Error in transcription: {e}")
        return []
    
'''Example from OpenAI Whisper Documentation - lower level model access and controls'''
def decode_audio(input_audio_file):
    try:
        model = whisper.load_model(model_size)

        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio("input_audio_file")
        audio = whisper.pad_or_trim(input_audio_file)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(input_audio_file).to(model.device)

        # detect the spoken language
        _, probs = model.detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}")

        # decode the audio
        options = whisper.DecodingOptions()
        result = whisper.decode(model, mel, options)

        # print the recognized text
        return result
    except Exception as e:
        logging.error(f"Error in decoding: {e}")
        return []
        


# Example usage (to be replaced with an actual audio file path)
# transcribe_audio("path_to_audio.wav")