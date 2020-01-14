from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import urllib.request
import io
import json
from transformers import pipeline
from flask import Flask, jsonify, request
from flask_restplus import Api, Resource, reqparse
from flask import abort

text_input = reqparse.RequestParser()
text_input.add_argument('text', required = True, help='Text to get a sentiment froma.')

app = Flask(__name__)
api = Api(app=app, version='1.0', title='Sentiment Analysis Service', description='A sentiment analysis service.')

@api.doc(params={})
@api.route('/cs/v1/nlp/sentimentanalysis')
class SentimentAnalysis(Resource):
 
  @api.expect(text_input)
  def post(self):
    args = text_input.parse_args()
    if not args['text']:
      abort(400, 'There is no text given.')

    if args['text'] == '':
      abort(400, 'Text parameter may be empty.')

    nlp = pipeline('sentiment-analysis')
    sentiments = nlp(args['text'])
    if len(sentiments) > 0:
      print(sentiments[0])
      return "{'label': '" + sentiments[0]['label'] + "', 'score':'" + str(sentiments[0]['score']) + "'}" 
    else:
      abort(400, "Could not detect a sentiment from sentence.")

if __name__ == '__main__':
  app.run(host='0.0.0.0', port='5000')

