import torch
from pathlib import Path 
import pandas as pd
import os
from tqdm import tqdm

from nltk.tokenize import sent_tokenize
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.models.sonar_text import (
    load_sonar_text_decoder_model,
    load_sonar_text_encoder_model,
    load_sonar_tokenizer,
)

device = torch.device("cuda")
t2enc = load_sonar_text_encoder_model("text_sonar_basic_encoder", device=device).eval()
text_tokenizer = load_sonar_tokenizer("text_sonar_basic_encoder")

embedder = TextToEmbeddingModelPipeline(t2enc, text_tokenizer, device=device)

filelist = sorted(list(Path("/disk1/data/wiki_727/train/").rglob("*")))
filelist = [i for i in filelist if os.path.isfile(i)]
parents = [i.parent.stem for i in filelist]
breakpoint()

print(filelist[:5], len(filelist))

languages = {"en": "english", "de":"german", "es":"spanish", "fr": "french", "pt":"portuguese"}
sonar_lang = {"en": "eng_Latn", "de":"deu_Latn", "es":"spa_Latn", "fr": "fra_Latn", "pt":"por_Latn"} # https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200

for file in tqdm(filelist, total=len(filelist)):
    out_path = Path("/disk1/data/preproc_sonar/wiki_727/train")/file.parent.stem
    out_path.mkdir(parents=True, exist_ok=True)
    if os.path.exists(out_path/f"{file.stem}.pt"):
        continue
    df = open(file, "r").readlines()
    df = [i for i in df if ("========" not in i or "**LIST**" not in i)]
    output = {
        "sentences": [],
        "embeddings": [],
        "labels": [],
    }
    for idx in tqdm(range(len(df["sourceItemMainText"])), leave=False):
        d = df.iloc[idx]
        sent = sent_tokenize(d['sourceItemMainText'], language="english")
        try:
            if len(sent)==0:
                continue
            with torch.no_grad():
                embeddings = embedder.predict(sent, source_lang=sonar_lang[lang], batch_size=64).detach().cpu()
        except:
            continue
        labs = [0]*(len(sent)-1) + [1]
        output['sentences'].append(sent)
        output["embeddings"].append(embeddings)
        output["labels"].extend(labs)
        output["keywords"].append(d['sourceItemKeywords'])
    output['embeddings'] = torch.cat(output['embeddings'])
    output['labels'] = torch.tensor(output['labels'])
    torch.save(output, out_path/f"{file.stem}.pt")