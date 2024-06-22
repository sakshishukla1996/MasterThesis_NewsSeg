import re
import json
from pytube import YouTube 
from pathlib import Path
from tqdm import tqdm

links = open("/disk1/projects/tagesschau/more_data_downloaded_links.txt").read().splitlines()
pattern = r"(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])"
out_path = Path("/data/tagesschau/more_data/")

for ilx, link in tqdm(enumerate(links), total=len(links)):
    try:
        link = re.search(pattern, link).group()
        yt = YouTube(link)
        yt.streams[0]
        captions = yt.captions
        output = captions['de'].json_captions
        json.dump(output, open(out_path / str(ilx) / "transcript.json", "w"))
        # print(out_path / str(ilx) / "transcript.json")
    except Exception as e: 
        print(e, link)
        continue


# import json
#     ...: links = open("/disk1/projects/tagesschau/more_data_downloaded_links.txt").read().splitlines()
#     ...: pattern = r"(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])"
#     ...:
#     ...: for ilx, link in tqdm(enumerate(links)):
#     ...:     link = re.search(pattern, link).group()
#     ...:     yt = YouTube(link)
#     ...:     yt.streams[0]
#     ...:     captions = yt.captions
#     ...:     output = captions['de'].json_captions
#     ...:     json
#     ...:     break
#     ...:     if len(list(captions.keys()))==0:
#     ...:         continue
#     ...:     else:
#     ...:         break