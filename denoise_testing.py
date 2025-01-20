import concurrent.futures
import json
import os
import time
import tracemalloc
import soundfile as sf

import requests

SAMPLE_RATE = 16000
SERVICE_URL = "http://localhost:8000/v1/denoise_filepath"
FILENAME = "clean_audio_static_noise_SNR0.wav"
DIRECTORY = "examples/"
FILEPATH = DIRECTORY + FILENAME
OUTPUT_DIRECTORY = "outputs/"


def ping_container():
    try:
        audio_bytes = {"file": open(FILEPATH, "rb")}
        response = requests.post(SERVICE_URL, files=audio_bytes)
        return response.status_code, response.text
    except requests.exceptions.RequestException as e:
        return None, str(e)


def output_denoised_audio(denoised_audio, directory):

    if not os.path.exists(directory):
        os.makedirs(directory)

    output_fp = OUTPUT_DIRECTORY + "denoised_" + FILENAME
        
    sf.write(output_fp, denoised_audio, 16000)


def main():
    tracemalloc.start()
    start_time = time.time()
    status_code, response_text = ping_container()

    print(f"Status Code: {status_code}")
    output_denoised_audio(json.loads(response_text)["denoise_audio"], OUTPUT_DIRECTORY)

    print(f"Completed in {time.time() - start_time} seconds")

    start_size, peak_use = tracemalloc.get_traced_memory()
    max_memory_use = peak_use - start_size

    print(f"Max Memory Allocation in (Bytes): {max_memory_use:.2f}B")
    print(f"Max Memory Allocation in (MB): {max_memory_use/(1024 * 1024):.2f}MB")

    tracemalloc.stop()


if __name__ == "__main__":
    main()
