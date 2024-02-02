import traceback
from .classification import DummyClassifier, BaseClassifierConfig, WhisperClassifierConfig, \
    WhisperClassifier
from .utils.diarization import draw_diarization
from .diarization import DummyDiarizer, DiarizationConfig, WhisperBasedDiarizationConfig, WhisperDiarizer, \
    EENDConfig, EENDDiarizer
from .transcription import DummyTranscriptor, WhisperTranscriptionConfig, WhisperTranscriptor
import gradio as gr
import os
from huggingface_hub import login

login(token=os.getenv('HF_TOKEN', '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'))

transcriptor = WhisperTranscriptor(WhisperTranscriptionConfig(whisper_checkpoint='anakib1/whisper-asr-0.1',
                                                              processor_checkpoint='anakib1/whisper-asr-0.1',
                                                              language='uk'))
diarizer = EENDDiarizer(EENDConfig(hf_api_model_path='anakib1/eend-sa',
                                   hf_api_model_name='model.pt',
                                   hf_api_processor_path='openai/whisper-small',
                                   max_num_speakers=2))
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
