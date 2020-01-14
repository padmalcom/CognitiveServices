from __future__ import absolute_import, division, print_function, unicode_literals

import io
import json
from flask import Flask, jsonify, request
from flask_restplus import Api, Resource, reqparse
from flask import abort
from summarizer import Summarizer

text_input = reqparse.RequestParser()
text_input.add_argument('text', required = True, help='The text to be summarized.')
text_input.add_argument('min_size', required = True, help='The minimum length of the summarized result.')

app = Flask(__name__)
api = Api(app=app, version='1.0', title='Text summarization Service', description='A service that sumarizes text.')

@api.doc(params={})
@api.route('/cs/v1/nlp/summarization')
class QuestionAnswering(Resource):
 
  @api.expect(text_input)
  def post(self):
    args = text_input.parse_args()
    if not args['text']:
      abort(400, 'There is no text to summarize.')

    if not args['min_size']:
      abort(400, 'Minimum size is not given.')

    if args['text'] == '' or args['min_size'] == '':
      abort(400, 'Text and min_size may not be empty.')

    ml = int(args['min_size'])
    model = Summarizer()
    result = model(args['text'], min_length=ml)
    full = ''.join(result)
    return full

if __name__ == '__main__':
  app.run(host='0.0.0.0', port='5000')

