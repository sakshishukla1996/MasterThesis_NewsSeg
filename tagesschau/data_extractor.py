from pytube import YouTube
from pathlib import Path

output_path_audio = Path(f"/data/tagesschau/audio")
output_path_description = Path(f"/data/tagesschau/descriptions")

file = open('yt_links.txt','r')
lines = file.readlines()
print("Found links: ", len(lines))
yt_links = [i.strip() for i in lines]
i = 1
for link in yt_links:
   
    print(f"Downloading from {link}")
    yt = YouTube(link)
    stream = yt.streams.get_by_itag(250)
    path_audio = output_path_audio / f"{i}"
    
    path_audio.mkdir(parents=True, exist_ok=True)
    print(f"Saving {link} to {path_audio}")

    stream.download(output_path=path_audio)
    description = yt.description
    description_path = output_path_description / f"{i}"
    description_path.mkdir(parents=True, exist_ok=True)
    
    with open(description_path / f"{i}.txt", "w") as f:
        f.write(description)
    i = i + 1
