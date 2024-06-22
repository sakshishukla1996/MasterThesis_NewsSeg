import os
from pathlib import Path
import gradio as gr
import shutil
from functools import partial
from transformers import pipeline
# from faster_whisper import WhisperModel

import numpy as np
import torch
import torchaudio

from sonar_functions import SonarSpeechPrediction, SonarTextPrediction
from plotting_library import plot_speech_segments, plot_text_segments

from io import BytesIO
import base64

device = torch.device("cuda:0")

pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        torch_dtype=torch.float16,
        device="cuda:0",
    )


# languages = {"en": "english", "de":"german", "es":"spanish", "fr": "french", "pt":"portuguese"}
# sonar_lang = {"en": "eng_Latn", "de":"deu_Latn", "es":"spa_Latn", "fr": "fra_Latn", "pt":"por_Latn"} # https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200


def iterative_transcriber(filepath, language):
    print(language)
    model_ckpt = "/data/euronews_dataset/weights/audio/multilingual_audio/all_all_all/all_all_all.ckpt"
    text_model_ckpt = "/data/euronews_dataset/weights/transcript/multilingual_text/all_all_all/all_all_all.ckpt"
    global pipe
    
    wav, sr = torchaudio.load(filepath)
    wav = wav[:1,:].reshape(-1).numpy()
    
    print(f'Input filepath is {filepath}')
    segments = pipe(filepath, chunk_length_s=30, batch_size=8, generate_kwargs={"task": "transcribe", "language": language}, return_timestamps="sentence", )
    print("Got all tokens. Running my model now!!")
    # for segment in segments['chunks']:
    #     all_text += f"([{segment['timestamp'][0]} -> {segment['timestamp'][1]}] {segment['text']}\n"
    #     yield [all_text, (sr, wav)]
    # Speech outputs
    sonarspeech = partial(SonarSpeechPrediction, device=torch.device("cuda:0"), model_ckpt=model_ckpt)
    predictions = sonarspeech(filepath, language=language)
    print(f"Got all the speech tokens now!!")
    img = plot_speech_segments(filepath, predictions)
    del sonarspeech
    print(img.size)


    buffered = BytesIO()
    # breakpoint()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # style="overflow-y: scroll; height:400px;
    html_code = "<div style='overflow-y: auto; max-height: 400px;'><img src='data:image/png;base64,{}'></div>".format(img_str)

    # Text outputs
    all_text = "\n".join([f"([{segment['timestamp'][0]} -> {segment['timestamp'][1]}] {segment['text']}\n" for segment in segments['chunks']])
    tokens = [segment['text'] for segment in segments['chunks']]
    sonartext = partial(SonarTextPrediction, device=torch.device("cuda:0"), model_ckpt=text_model_ckpt)
    predictions_text = sonartext(tokens, language=language)
    return [predictions_text, (sr, wav), html_code]

demo = gr.Interface(
    fn=iterative_transcriber,
    inputs=[
        "text",
        gr.Dropdown(
            ["en", "de", "hi"], label="Language", info="We currently have three encoders working finer than others."
        )
    ],
    outputs=[
        gr.HighlightedText(
            label="Text segments",
            show_legend=True,
            color_map={"1": "red", "2": "green", "3": "blue", "4": "yellow", "5": "gray", "0": "orange"}),
        "audio",
        "html"
    ]
)


# interface = gr.TabbedInterface([demo, demo2], tab_names=['End-to-End', "Pipeline"])

if __name__=="__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7757
    )