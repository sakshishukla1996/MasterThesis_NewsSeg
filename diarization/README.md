## Quick Usage
```bash
python main.py --input-file <input_wav_file> --output-dir <output_directory>
```
where
```
--input-file: Path to the input WAV file (16 kHz, mono-channel, and 16-bit PCM).
--output-dir: Path to the output directory where the diarization results will be saved.
```

After running the script, it will perform diarization on the input audio file, extract segments, compute embeddings, and generate the RTTM and JSONL files containing the diarization results. The diarization steps will be logged to the console, and a success message will be displayed for each step along with the elapsed time.

## Details
Install the required packages using,

```bash
$ pip install -r requirements.txt
```

then use `Pipeline` to start diarization process,

```python
>>> from main import Pipeline
>>> diarizer = Pipeline.init_from_wav("./test/tagesschau02092019.wav")
>>> segments_file = diarizer.write_to_segments("./output_folder")
```

This will use input `tagesschau02092019.wav` file (sampled at 16 kHz) and returns full path of the JSONL-formatted output with other experimental files into the `output_folder/`.

## Output Files

|                                  | Description                                                    |
| :------------------------------: | -------------------------------------------------------------- |
| `./output_folder/test_file.ark`  | Speaker embedding vectors stored in Kaldi's ".ark" file format |
| `./output_folder/test_file.seg ` | Extraction segments indicating duration of speaker embeddings  |
| `./output_folder/test_file.rttm` | Speaker diarization results for the input file in RTTM format  |
| `./output_folder/test_file.json` | Sorted output in JSONL format for better parsing/visualization |