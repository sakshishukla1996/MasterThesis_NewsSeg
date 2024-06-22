from io import BytesIO
import torchaudio
import torch
from matplotlib import pyplot as plt
import math
import random
from PIL import Image

def plot_speech_segments(file, segment_labels):
    buffer=BytesIO()
    
    wav, sr = torchaudio.load(file)
    wav = wav[:1, :]
    if sr!=16000:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16000)
        sr=16000

    duration = wav.shape[-1]/sr
    # breakpoint()
    fig = plt.figure(figsize=(100,4))
    time_axis = torch.linspace(0, duration, steps=wav.size(1))

    plt.plot(time_axis.numpy(), wav.flatten().numpy())
    plt.xticks(range(0, int(duration) + 1, 10))
    segments = [(i*10, (i+1)*10) for i in range(math.ceil(duration/10))]
    color="crimson"
    for (start, end), label in zip(segments, segment_labels):
        if label==int(1):
            color = random.choice(['red','green','blue','orange','yellow','gray'])
        plt.axvspan(start, end, color=color, alpha=0.3)
        # plt.axvline(x=end, color='b', linestyle='--', alpha=0.2)

    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    img = Image.open(buffer)
    return img

def plot_text_segments(file, segment_labels):
    buffer=BytesIO()
    
    wav, sr = torchaudio.load(file)
    wav = wav[:1, :]
    if sr!=16000:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16000)
        sr=16000
    duration = wav.shape[-1]/sr
    fig = plt.figure(figsize=(100,4))
    time_axis = torch.linspace(0, duration, steps=wav.size(1))

    plt.plot(time_axis.numpy(), wav.flatten().numpy())
    plt.xticks(range(0, int(duration) + 1, 10))
    segments = [(i*10, (i+1)*10) for i in range(math.ceil(duration/10))]
    color="crimson"
    for (start, end), label in zip(segments, segment_labels):
        if label==int(1):
            color = random.choice(['red','green','blue','orange','yellow','gray'])
        plt.axvspan(start, end, color=color, alpha=0.3)
        # plt.axvline(x=end, color='b', linestyle='--', alpha=0.2)

    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    img = Image.open(buffer)
    return img