import torch
from pathlib import Path
from tqdm import tqdm

inppath = Path("/data/preproc_sonar/dw_2019/de")  # es,de,pt,en,fr
files = list(inppath.iterdir())
OUT_ROOT = "/data/preproc_sonar_single/"
for file in tqdm(files):
    out_root = Path(OUT_ROOT) / file.parent.stem
    a = torch.load(file)
    running_pointer=0
    for isx, sent in tqdm(enumerate(a['sentences']), total=len(a['sentences']), leave=False):
        thresh = len(sent)
        emb = a['embeddings'][running_pointer:running_pointer+thresh]
        labs = a['labels'][running_pointer:running_pointer+thresh]
        running_pointer+=thresh
        # print(thresh, emb.shape, labs.shape, out_root / f"{file.stem}_{isx}.pt")
        new_dict = {
            'sentences': sent,
            'language': a['language'][isx],
            'embeddings': emb.clone().cpu(),
            'labels': labs.cpu(),
            'keywords': a['keywords'][isx] 
        }
        if thresh==1:
            breakpoint()
        # torch.save(new_dict, out_root / f"{file.stem}_{isx}.pt")
        # if running_pointer>100:
        #     break