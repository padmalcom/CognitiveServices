import io
import json
from flask import Flask, jsonify, request
from flask_restplus import Api, Resource
import spacy
import en_core_web_lg
import en_core_web_sm
from flask import abort

app = Flask(__name__)
api = Api(app=app, version='1.0', title='A simple NER service.', description='This service provides functions to extract entities from text.')

@api.doc(params={'model': 'Either a small or a large nlp model.', 'text': 'The text to parse.'})
@api.route('/cs/v1/nlp/entityextraction/<string:model>/<string:text>')
class entityextraction(Resource):
  def get(self, model, text):

    if text == "":
      abort(400, 'Please provide a non-empty text.')

    if not model == "large" and not model == "small":
      abort(400, 'Model must be either \'small\' or \'large\'.')

    # do ner
    print("Processing text " + text)
    if model == "small":
      nlp = en_core_web_sm.load()
    else:
      nlp = en_core_web.lg.load()
    doc = nlp(text)

    result = []
    for ent in doc.ents:
      result.append({"text":ent.text, "start":ent.start_char, "end":ent.end_char, "label":ent.label_})
      print(ent.text, ent.start_char, ent.end_char, ent.label_)
    return jsonify(result)

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000)
