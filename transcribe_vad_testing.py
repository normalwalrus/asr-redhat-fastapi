import concurrent.futures
import json
import os
import time
import tracemalloc

import requests

SAMPLE_RATE = 16000
SERVICE_URL = "http://localhost:8001/v1/transcribe_resample_diarize_filepath"

# FILENAME = "GMT20250113-075636_Recording.wav"
# DIRECTORY = "/home/digitalhub/Desktop/ian_projects/HR/audio/"

FILENAME = "steroids_120sec.wav"
DIRECTORY = "examples/wav/"
FILEPATH = DIRECTORY + FILENAME

#FILEPATH = "/home/digitalhub/Desktop/ian_projects/CIO-transcribe/redhat-diarizer-v5.0/examples/mp3/steroids_120sec.mp3"
OUTPUT_DIRECTORY = "outputs/"


def ping_container():
    try:
        audio_bytes = {"file": open(FILEPATH, "rb")}
        response = requests.post(SERVICE_URL, files=audio_bytes)
        print(response)
        return response.status_code, response.text
    except requests.exceptions.RequestException as e:
        return None, str(e)


def output_transcriptions(transcription, directory):

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(OUTPUT_DIRECTORY + "Output_" + FILENAME + ".txt", "w") as text_file:
        text_file.write(transcription)


def main():
    tracemalloc.start()
    start_time = time.time()
    status_code, response_text = ping_container()

    print(f"Status Code: {status_code}, Response: {response_text}")
    output_transcriptions(json.loads(response_text)["transcription"], OUTPUT_DIRECTORY)

    print(f"Completed in {time.time() - start_time} seconds")

    start_size, peak_use = tracemalloc.get_traced_memory()
    max_memory_use = peak_use - start_size

    print(f"Max Memory Allocation in (Bytes): {max_memory_use:.2f}B")
    print(f"Max Memory Allocation in (MB): {max_memory_use/(1024 * 1024):.2f}MB")

    tracemalloc.stop()


if __name__ == "__main__":
    main()
