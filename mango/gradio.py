import traceback
from .classification import DummyClassifier, BaseClassifierConfig, WhisperClassifierConfig, \
    WhisperClassifier, WhisperClassifierEx, WhisperClassifierExConfig
from .utils.diarization import draw_diarization
from .diarization import DummyDiarizer, DiarizationConfig, WhisperBasedDiarizationConfig, WhisperDiarizer, \
    EENDConfig, EENDDiarizer
from .transcription import DummyTranscriptor, WhisperTranscriptionConfig, WhisperTranscriptor
import gradio as gr
import os
from huggingface_hub import login

login(token=os.getenv('HF_TOKEN', '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'))
transcriptor = WhisperTranscriptor(WhisperTranscriptionConfig(whisper_checkpoint='Yehor/whisper-small-ukrainian',
                                                              processor_checkpoint='Yehor/whisper-small-ukrainian',
                                                              language='uk'))
diarizer = EENDDiarizer(EENDConfig(hf_api_model_path='anakib1/eend-sa',
                                   run_id='run-24-02-06.20-17',
                                   hf_api_model_name='model.pt',
                                   hf_api_processor_path='openai/whisper-small',
                                   max_num_speakers=3))

classifier = WhisperClassifierEx(WhisperClassifierExConfig(id2label={0: 'dog_bark',
                                                                     1: 'children_playing',
                                                                     2: 'car_horn',
                                                                     3: 'air_conditioner',
                                                                     4: 'street_music',
                                                                     5: 'gun_shot',
                                                                     6: 'siren',
                                                                     7: 'engine_idling',
                                                                     8: 'jackhammer',
                                                                     9: 'drilling'}))


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
