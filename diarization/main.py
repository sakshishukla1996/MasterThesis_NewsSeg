#!/usr/bin/env python

# Author: Tugtekin Turan
# E-Mail: tugtekin.turan@iais.fraunhofer.de
# Date: 2023-05-19
# Description: Main file for the speaker diarization process

import os
import sys
import json
import shutil
import logging
import time
from pathlib import Path

# set the log severity level to "FATAL"
logging.basicConfig(level=logging.FATAL)

# append directories to the PYTHONPATH
abspath = os.path.dirname(os.path.abspath(__file__))
local_path = os.path.join(abspath, "local")
utils_path = os.path.join(abspath, "utils")
sys.path += [local_path, utils_path]

from create_vad_segments import VAD
from AHC_with_vbHMM import Diarization
from extract_embeddings import EmbeddingExtraction


class Pipeline:
    def __init__(self, input_file, max_speakers=None):
        self.input_file = Path(input_file)
        self.max_speakers = max_speakers
        self.segments = VAD()
        self.extractor = EmbeddingExtraction()
        self.diarizer = Diarization()

    def diarize(self, output_dir):
        """Orchestration method that performs diarization on the input audio file"""
        output_path = self._prepare_output_directory(output_dir)

        start_time_1 = time.monotonic()
        segments = self.extract_segments(output_path)
        self._print_success_message(1, "Calculating segments", self._compute_elapsed_time(start_time_1))

        start_time_2 = time.monotonic()
        embeddings = self.compute_embeddings(segments, output_path)
        self._print_success_message(2, "Embedding extraction", self._compute_elapsed_time(start_time_2))

        ark_file, seg_file = self._write_output_files(embeddings, output_path)

        start_time_3 = time.monotonic()
        rttm_path = self.run_diarization(ark_file, seg_file, output_path)
        self._print_success_message(3, "Speaker segmentation", self._compute_elapsed_time(start_time_3))

        print(f"[OK] All diarization steps were finished! Total wall time = {self._compute_elapsed_time(start_time_1)}")
        return str(rttm_path)

    def extract_segments(self, output_dir):
        """Extracts segments from the input audio file using VAD module"""
        return self.segments.extract(str(self.input_file), str(output_dir))

    def compute_embeddings(self, segments, output_dir):
        """Computes embeddings from the VAD segments using an embedding extraction model"""
        label_file = output_dir.joinpath("vad_files")
        return self.extractor.compute_embeddings(str(self.input_file), str(label_file))

    def run_diarization(self, ark_file, seg_file, output_dir):
        """Runs the diarization algorithm over the embeddings and VAD segments"""
        self.diarizer.process(str(output_dir), str(ark_file), str(seg_file), self.max_speakers)
        return output_dir.joinpath(f"{self.input_file.stem}.rttm")

    def _write_output_files(self, embeddings, output_dir):
        """Write segment files and embedding features in formats suitable for diarization"""
        file_name = self.input_file.stem
        ark_file = output_dir.joinpath(f"{file_name}.ark")
        seg_file = output_dir.joinpath(f"{file_name}.seg")
        EmbeddingExtraction.write_embeddings_to_ark(ark_file, embeddings)
        EmbeddingExtraction.write_segments_to_txt(seg_file, embeddings, str(self.input_file))
        return ark_file, seg_file

    def _prepare_output_directory(self, output_dir):
        output_dir = Path(output_dir)
        if output_dir.exists():
            for item in output_dir.iterdir():
                if item.is_dir() and item.name == "vad_files":
                    shutil.rmtree(item)
                elif item.is_file():
                    if item.suffix in [".ark", ".json", ".rttm", ".seg"]:
                        item.unlink()
        else:
            output_dir.mkdir(parents=True)
        return output_dir

    def _compute_elapsed_time(self, start_time):
        end_time = time.monotonic()
        elapsed_time = end_time - start_time
        return time.strftime("%M:%S", time.gmtime(elapsed_time))

    def _print_success_message(self, step_number, process_name, elapsed_time):
        print(f"Step {step_number}/3: {process_name} completed successfully. Elapsed time = {elapsed_time}")

    def _rttm_to_jsonl(self, input_path):
        output_file = os.path.splitext(input_path)[0] + ".json"

        with open(input_path, "r") as file:
            data = file.readlines()

        id_mapping = {}
        current_speaker_id = 1

        jsonl_data = []
        for line in data:
            line_data = line.split()
            if line_data[7] not in id_mapping:
                id_mapping[line_data[7]] = "spk_" + str(current_speaker_id)
                current_speaker_id += 1

            segment = {}
            segment["Start"] = "{:.2f}".format(round(float(line_data[3]), 2))
            segment["End"] = "{:.2f}".format(round(float(line_data[3]) + float(line_data[4]), 2))
            segment["Speaker"] = id_mapping[line_data[7]]
            jsonl_data.append(segment)

        with open(output_file, "w") as file:
            for item in jsonl_data:
                file.write(json.dumps(item) + "\n")

    def write_to_segments(self, output_dir):
        output_dir = Path(output_dir)
        rttm_path = self.diarize(output_dir)
        self._rttm_to_jsonl(rttm_path)
        print(f"--> Please check the results generated at {os.path.splitext(rttm_path)[0]}.json")
        return str(os.path.splitext(rttm_path)[0] + ".json")

    @classmethod
    def init_from_wav(cls, input_file, max_speakers=None):
        return cls(input_file, max_speakers)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run speaker diarization")
    parser.add_argument("--input-file", type=str, required=True, help="path to input WAV file")
    parser.add_argument("--output-dir", type=str, required=True, help="path to output directory")
    parser.add_argument("--max-speakers", type=int, default=None, help="optional max. number of speakers")
    args = parser.parse_args()

    diarizer = Pipeline.init_from_wav(args.input_file, args.max_speakers)
    diarizer.write_to_segments(args.output_dir)


if __name__ == "__main__":
    main()
    # remove stored bytecode compiled versions
    for dirpath, dirnames, _ in os.walk(os.getcwd(), topdown=False):
        if "__pycache__" in dirnames:
            pycache_path = os.path.join(dirpath, "__pycache__")
