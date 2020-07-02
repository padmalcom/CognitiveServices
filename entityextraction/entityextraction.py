import io
import json
from flask import Flask, jsonify, request
from flask_restplus import Api, Resource, reqparse
import spacy
import en_core_web_lg
import en_core_web_sm
from flask import abort

app = Flask(__name__)
api = Api(app=app, version='1.0', title='A simple NER service.', description='This service provides functions to extract entities from text.')

input_parameters = reqparse.RequestParser()
input_parameters.add_argument(name="text", location="form", required=True, help="The text to analyze.")
input_parameters.add_argument(name="model", location="form", required=True, help="Choose between a small (fast) and a large (better detection) model.", choices=("small","large"))

@api.doc(params={})
@api.route('/cs/v1/nlp/entityextraction', methods=["POST"])
class entityextraction(Resource):

  @api.expect(input_parameters)
  def post(self):
  
    args = input_parameters.parse_args()
    text = args['text']
    model = args['model']

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
