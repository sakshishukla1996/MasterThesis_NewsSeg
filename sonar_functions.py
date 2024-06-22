from itertools import cycle
import sys
sys.path.append("/data/projects/projects/sonar_multilingual_segmentation/")

import torch
import torchaudio
import random

from sonar.inference_pipelines.speech import SpeechToEmbeddingModelPipeline
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

device= torch.device("cuda")

languages = {"en": "english", "de":"german", "es":"spanish", "fr": "french", "pt":"portuguese"}
sonar_lang = {"en": "eng_Latn", "de":"deu_Latn", "es":"spa_Latn", "fr": "fra_Latn", "pt":"por_Latn", "hi": "hin_Deva"}
# https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200

def SonarSpeechPrediction(filepath, language, device, model_ckpt):
    encoders = {
        "en": "sonar_speech_encoder_eng",
        "de": "sonar_speech_encoder_deu",
        "es": "sonar_speech_encoder_spa",
        "fr": "sonar_speech_encoder_fra",
        "pt": "sonar_speech_encoder_por",
        "hi": "sonar_speech_encoder_hin"
    }
    s2vec_model = SpeechToEmbeddingModelPipeline(encoder=encoders[language], device=device)

    model = torch.load(model_ckpt)
    net = model['hyper_parameters']["net"]
    net.load_state_dict(model['state_dict'], strict=False)
    net.to(device)
    net.eval()

    wav, sr = torchaudio.load(filepath)
    wav = wav[:1, :]
    if sr!=16000:
        audio_low = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16000)    
    else:
        audio_low = wav
    sr=16000
    audio_10_splits = list(torch.split(audio_low, split_size_or_sections=(16_000 * 10), dim=-1))

    with torch.no_grad():
        embeddings = s2vec_model.predict(audio_10_splits, batch_size=16)
        output_10 = torch.nan_to_num(embeddings, nan=0.0)

        prepad = torch.cat([torch.zeros(2, output_10.shape[-1], device=output_10.device), output_10])
        currpad = torch.cat([torch.zeros(1, output_10.shape[-1], device=output_10.device), output_10, torch.zeros(1, output_10.shape[-1], device=output_10.device)])
        nextpad = torch.cat([output_10, torch.zeros(2, output_10.shape[-1], device=output_10.device)])
        inp = torch.cat([prepad, currpad, nextpad], dim=-1)[:-2]  # 100

        logits = net(inp.unsqueeze(0)).detach().cpu()
    preds = logits.sigmoid().argmax(-1).squeeze(0)
    print(f"Got the following predictions from the speech class: {preds}")

    return preds.numpy()


def SonarTextPrediction(tokens, language, device, model_ckpt):
    global sonar_lang
    t2vec_model = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder", device=device)

    model = torch.load(model_ckpt)
    net = model['hyper_parameters']["net"]
    net.load_state_dict(model['state_dict'], strict=False)
    net.to(device)
    net.eval()
    # Tokenize the text.

    with torch.no_grad():
        embeddings = t2vec_model.predict(tokens, batch_size=16, source_lang=sonar_lang[language])
        output_10 = torch.nan_to_num(embeddings, nan=0.0)

        prepad = torch.cat([torch.zeros(2, output_10.shape[-1], device=output_10.device), output_10])
        currpad = torch.cat([torch.zeros(1, output_10.shape[-1], device=output_10.device), output_10, torch.zeros(1, output_10.shape[-1], device=output_10.device)])
        nextpad = torch.cat([output_10, torch.zeros(2, output_10.shape[-1], device=output_10.device)])
        inp = torch.cat([prepad, currpad, nextpad], dim=-1)[:-2]  # 100

        logits = net(inp.unsqueeze(0)).detach().cpu()
    preds = logits.sigmoid().argmax(-1).squeeze(0).numpy()
    print(f"Got the following predictions from the text class: {preds}")
    outputs = []
    seg = 1
    for tok, pred in zip(tokens, preds):
        if pred==1:
            seg +=1
            seg=seg%5
        outputs.append((tok, str(seg)))
        outputs.append(("\n", str(seg)))
    return outputs