import traceback
from mango.classification import DummyClassifier, BaseClassifierConfig, WhisperClassifierConfig, \
    WhisperClassifier
from mango.diarization import DummyDiarizer, draw_diarization, DiarizationConfig, WhisperBasedDiarizationConfig, \
    WhisperDiarizer
from mango.transcription import DummyTranscriptor, WhisperTranscriptionConfig, WhisperTranscriptor
import gradio as gr

safe = None

transcriptor = WhisperTranscriptor(WhisperTranscriptionConfig(whisper_checkpoint='anakib1/whisper-small-uk',
                                                              processor_checkpoint='openai/whisper-small'))
diarizer = WhisperDiarizer(WhisperBasedDiarizationConfig(whisper_checkpoint='anakib1/whisper-diarization-0.2',
                                                         processor_checkpoint='openai/whisper-tiny',
                                                         max_num_speakers=3))
classifier = WhisperClassifier(WhisperClassifierConfig(whisper_checkpoint='anakib1/whisper-tiny-urban',
                                                       processor_checkpoint='openai/whisper-tiny'))


def describe_audio(mic=None, file=None):
    try:
        if mic is not None:
            sr, audio = mic
        elif file is not None:
            sr, audio = file
        else:
            return "You must either provide a mic recording or a file", None, None

        if len(audio.shape) > 1:
            audio = audio[0]

        transciption = transcriptor.transcribe(audio, sr)
        diarization = diarizer.diarize(audio, sr)
        diarization_pic = draw_diarization(diarization)
        classification_result = classifier.classify(audio, sr)

        return transciption, diarization_pic, classification_result

    except Exception as ex:
        return traceback.format_exc(), None, None


gradio_interface = gr.Interface(
    fn=describe_audio,
    inputs=gr.Audio(sources=["microphone", "upload"], type="numpy"),
    outputs=[gr.Textbox(label='transcription'), gr.Image(label='diarization', height=400, width=800),
             gr.Label(label='noise context', num_top_classes=4)],
    allow_flagging='never'
)
