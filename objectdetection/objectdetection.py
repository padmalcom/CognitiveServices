import io
import os
import json
from flask import Flask, jsonify, request
from flask_restplus import Api, Resource, reqparse
from flask import abort
import cv2
import cvlib
import werkzeug
import numpy as np
import base64

# You can change this to any folder on your system
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

file_upload = reqparse.RequestParser()
file_upload.add_argument('img_file', type=werkzeug.datastructures.FileStorage, location='files',  required=True,  help='Image file')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)
api = Api(app=app, version='1.0', title='Object detection service', description='This service detects objects in an image and returns the bounding box and labels.')

@api.doc(params={})
@api.route('/cs/v1/vision/objectdetection')
class objectdetection(Resource):

  def detect_objects_in_image(self, file_stream):
    filestr = file_stream.read()
    npimg = np.fromstring(filestr, np.uint8)
    im = cv2.imdecode(npimg,cv2.IMREAD_COLOR)

    bbox, label, conf = cvlib.detect_common_objects(im)
    results = []
    for b,l,c in zip(bbox, label, conf):
      results.append("{dimensions:("+str(b[0])+"," + str(b[1]) + "," + str(b[2]) + "," + str(b[3]) + "), label:" + l + ", confidence:" + str(c) + "}")
    return jsonify(results)
    
  @api.expect(file_upload)
  def post(self):
  
    args = file_upload.parse_args()
    
    if not args['img_file']:
      abort(400, 'Please add a file to the post parameters.')
      
    if args['img_file'].mimetype != 'image/jpeg' and args['img_file'].mimetype != 'image/png':
      abort(400, 'Image must be a jpg or a png.')

    file = args['img_file']

    if file.filename == '':
      abort(400, 'Please provide a non-empty file name.')
      
    if not allowed_file(file.filename):
      abort(400, 'The file type provided is not allowed.') 

    if file:
      return self.detect_objects_in_image(file)

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000)
