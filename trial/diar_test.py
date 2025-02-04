from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from nemo.utils import nemo_logging
import logging

logger_nemo = logging.getLogger('nemo_logger')
logger_nemo.disabled = True

Diarizer = NeuralDiarizer.from_pretrained("/opt/app-root/pretrained_models/diar_msdd_telephonic.nemo").to('cuda')
print(Diarizer.diarizer.collar)