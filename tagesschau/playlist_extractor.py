from pytube import YouTube, Playlist
from pathlib import Path

playlist = Playlist(
    # "https://www.youtube.com/watch?v=PZzeJgV7jbI&list=PL4A2F331EE86DCC22"
    "https://www.youtube.com/playlist?list=PLcDghvQhYD9IiUf2hOKJlAh3UYzVHiJjR"
)
playlist_videos = playlist.video_urls
output_path_audio = Path("/data/akashvani/orig_videos")

for i, video in enumerate(playlist_videos):
    try:
        yt = YouTube(video)
        # if i==0:
            # breakpoint()
        if i==150:
            break
        stream = yt.streams.get_by_itag(139)
        path_audio = output_path_audio / f"{i}"
        path_audio.mkdir(parents=True, exist_ok=True)
        # captions = yt.captions
        stream.download(output_path=path_audio)
        description = yt.description
        with open(path_audio / f"{i}.txt", "w") as f:
            f.write(description)
        # with open(path_audio / f"{i}_captions.json", "w") as f:
            # f.write(captions)
        print(f"Saving {video}")
    except Exception as e:
        print(f"Failed to download {video} because of {e}")


from pytube import YouTube

# yt = YouTube("https://www.youtube.com/watch?v=9rsCFUH9fGI")
# stream = yt.streams[0]
# output = stream.download(output_path="./")
# captions = yt.captions
# >>> {}
