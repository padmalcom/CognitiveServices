import io
import json
from flask import Flask, jsonify, request
from flask_restplus import Api, Resource
from flask import abort
import torch

app = Flask(__name__)
api = Api(app=app, version='1.0', title='A simple neuronal translation service.', description='This service provides functions to translate text between German, English, Russian and French.')

@api.doc(params={'langpair': 'The language pair (en2fr, en2de, de2en, en2ru, ru2en)', 'text': 'The text to translate.'})
@api.route('/cs/v1/nlp/translate/<string:langpair>/<string:text>')
class translation(Resource):
  def get(self, langpair, text):
    #parameters
    if text == "":
      abort(400, 'Please provide a non-empty text.')

    langpairs = {'en2fr':'transformer.wmt14.en-fr', 'en2de':'transformer.wmt19.en-de', 'de2en':'transformer.wmt19.de-en', 'en2ru':'transformer.wmt19.en-ru', 'ru2en':'transformer.wmt19.ru-en'}

    if not langpair in langpairs.keys():
      abort(400, 'No valid language pair defined.')
    
    modelname = langpairs.get(langpair)

    print("Language pair is " + langpair + " and model is " + modelname)

    model = torch.hub.load('pytorch/fairseq', modelname, tokenizer='moses', bpe='fastbpe')

    result = model.translate(text, beam=5)
    return jsonify(result)

if __name__ == '__main__':
  app.run(host='0.0.0.0', port='5000')
