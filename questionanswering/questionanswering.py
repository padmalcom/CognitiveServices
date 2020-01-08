from __future__ import absolute_import, division, print_function, unicode_literals

from tqdm import trange
import torch
import torch.nn.functional as F
import numpy as np
from transformers import pipeline
import io
import json
from flask import Flask, jsonify, request
from flask_restplus import Api, Resource, reqparse
from flask import abort

text_input = reqparse.RequestParser()
text_input.add_argument('context', required = True, help='A context text.')
text_input.add_argument('question', required = True, help='The question to answer.')

app = Flask(__name__)
api = Api(app=app, version='1.0', title='Question Answering Service', description='A service that answers questions based on a given context.')

@api.doc(params={})
@api.route('/cs/v1/nlp/questionanswering')
class QuestionAnswering(Resource):
 
  @api.expect(text_input)
  def post(self):
    args = text_input.parse_args()
    if not args['context']:
      abort(400, 'There is no context given.')

    if not args['question']:
      abort(400, 'There is no question to answer.')

    if args['question'] == '' or args['context'] == '':
      abort(400, 'Neither question nor context may be empty.')

    nlp = pipeline('question-answering')
    return nlp({'question': args['question'], 'context': args['context']})

if __name__ == '__main__':
  app.run(host='0.0.0.0', port='5000')

