import io
import json
from flask import Flask, jsonify, request
from flask_restplus import Api, Resource
from flask import abort, send_from_directory
import torch
import os

import numpy as np
from scipy.io.wavfile import write

app = Flask(__name__)
api = Api(app=app, version='1.0', title='A simple text-to-speech service.', description='This service provides functions to convert text to speech.')

UPLOAD_DIRECTORY = "/tmp"

@api.doc(params={'text': 'The text to convert to wav.'})
@api.route('/cs/v1/voice/texttospeech/<string:text>')
class texttospeech(Resource):
  def get(self, text):
    #parameters
    if text == "":
      abort(400, 'Please provide a non-empty text.')
      
    if not torch.cuda.is_available():
      abort(400, 'This service requires cuda which is not found on the executing machine.')

    # do tts
    print("Processing text " + text)
    tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2', map_location=torch.device('cpu'))

    tacotron2 = tacotron2.to('cpu')
    tacotron2.eval()

    waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow', map_location=torch.device('cpu'))
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow = waveglow.to('cpu')
    waveglow.eval()

    # preprocessing
    sequence = np.array(tacotron2.text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.from_numpy(sequence).to(device='cuda', dtype=torch.int64)

    # run the models
    with torch.no_grad():
      _, mel, _, _ = tacotron2.infer(sequence)
      audio = waveglow.infer(mel)
    audio_numpy = audio[0].data.cpu().numpy()
    rate = 22050

    write("/tmp/audio.wav", rate, audio_numpy)
    return send_from_directory(UPLOAD_DIRECTORY, "audio.wav", as_attachment=True)

if __name__ == '__main__':
  if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)
  app.run(host='0.0.0.0', port=5000)
