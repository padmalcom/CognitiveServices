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
import wikipedia

text_input = reqparse.RequestParser()
text_input.add_argument('context', required = True, help='A context text.')
text_input.add_argument('question', required = True, help='The question to answer.')

wiki_input = reqparse.RequestParser()
wiki_input.add_argument('wiki', required = True, help='A topic in wikipedia.')
wiki_input.add_argument('question', required = True, help='The question to answer.')


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

@api.doc(params={})
@api.route('/cs/v1/nlp/questionanswering/wikipedia')
class QAWikipedia(Resource):

  @api.expect(wiki_input)
  def post(self):
    args = wiki_input.parse_args()
    if not args['wiki']:
      abort(400, 'There is no wikipedia topic given.')

    if not args['question']:
      abort(400, 'There is no question to answer.')

    if args['question'] == '' or args['wiki'] == '':
      abort(400, 'Neither question nor wikipedia context may be empty.')

    wiki_article = wikipedia.search(args['wiki'])
    print("Suggestion for " + args['wiki'] + " is " + str(len(wiki_article)))
    if len(wiki_article) > 0:
      print("Using topic " + wiki_article[0])
      wiki_content = wikipedia.summary(wiki_article[0])

      nlp = pipeline('question-answering')
      return nlp({'question': args['question'], 'context': wiki_content})

    #abort(400, 'No context found for ' + args['wiki'])
if __name__ == '__main__':
  app.run(host='0.0.0.0', port='5000')

