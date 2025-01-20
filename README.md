ENSURE YOU HAVE DOCKER INSTALLED

To create docker image:

1. Open up terminal here
2. docker load --input whisper-asr-fastapi-service.tar

Spin up docker container with this image:
(ENSURE YOU HAVE CREATED DOCKER IMAGE)

1. Open up terminal
2. docker-compose up asr-service

Testing out docker asr transcription:

1. Install Python (If you do not have Python)
2. Install the requests library (pip install requests)
3. Open up the transcribe_vad_testing.py file
   a. Ensure Filename of audio file is correct (ideally put a .wav file)
4. run testing.py

Testing out denoiser/amplification function:

1. Install Python (If you do not have Python)
2. Install the requests library (pip install requests)
3. Open up the denoise_testing.py file
   a. Ensure Filename of audio file is correct (ideally put a .wav file)
4. run denoise_testing.py
5. Denoised/amplified audio will appear in outputs folder

API endpoints:

http://localhost:8000/v1/transcribe_filepath -> transcribe based on filepath
http://localhost:8000/v1/transcribe_diarize_filepath -> transcribe and diarize based on filepath
http://localhost:8000/v1/denoise_filepath -> denoise/amplify based on filepath

.env.dev file:

PRETRAINED_MODEL_DIR: DO NOT CHANGE (should be "/opt/app-root/pretrained_models/whisper-large-v3")
PRETRAINED_DIAR_MODEL_DIR: DO NOT CHANGE (should be "/opt/app-root/pretrained_models/diar_msdd_telephonic.nemo")
SAMPLE_RATE: DO NOT CHANGE (should be 16000)

------------------------ FOR TRANSCRIPTION -----------------------------------------
DEVICE: Can choose between 'cpu' or 'cuda' (GPU), if not sure, can write any string to automatically choose based on your machine
TIMESTAMPS_FORMAT: Choose between 'seconds' or 'minutes' (defaults to 'seconds')

------------------------ FOR DENOISING/AMPLIFICATION -----------------------------------------
DRY: Choose a floating point number from 0 to 1 (the closer the number is to 0, the stronger the denoiser)
AMPLIFICATION_FACTOR: Choose a number to amplify the audio by (choose 1 to use the original audio)


Add in the pretrained_models folder to get this to work