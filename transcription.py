import whisper
import logging
import configparser

# Load configurations
config = configparser.ConfigParser()
config.read('config.ini')

# Load the Whisper model based on configuration
model_size = config.get('Whisper', 'ModelSize')
model = whisper.load_model(model_size)

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
    verbose = True
    audio=input_audio_file

    try:
        # Transcribe the audio
        result = model.transcribe(audio, verbose)
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