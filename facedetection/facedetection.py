import io
import os
import json
import werkzeug
from flask import Flask, jsonify, request
from flask_restplus import Api, Resource, reqparse
from flask import abort
import face_recognition
import glob
import numpy as np

# You can change this to any folder on your system
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

file_upload = reqparse.RequestParser()
file_upload.add_argument('img_file', type=werkzeug.datastructures.FileStorage, location='files',  required=True,  help='Image file')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)
api = Api(app=app, version='1.0', title='Face detection service.', description='This service detects faces in an image and returns the bounding box and a name of the person if known.')

@api.doc(params={})
@api.route('/cs/v1/vision/facedetection')
class facedetection(Resource):

  def detect_faces_in_image(self, file_stream):
  
    known_face_encodings = []
    known_face_names = []
    
    for f in os.listdir(os.path.join(os.getcwd(), "known")):
      if f.endswith(".jpg"):
        known_img = face_recognition.load_image_file(os.path.join(os.path.join(os.getcwd(), "known"), f))
        if len(face_recognition.face_encodings(known_img)) > 0:
          known_img_enc = face_recognition.face_encodings(known_img)[0]
          name = os.path.splitext(os.path.basename(f))[0]
          known_face_encodings.append(known_img_enc)
          known_face_names.append(name)
        
    img = face_recognition.load_image_file(file_stream)
    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

    results = []
    for (top, right, bottom, left), uf in zip(face_locations, unknown_face_encodings):
      matches = face_recognition.compare_faces(known_face_encodings, uf)
      name = "[unknown]"
      face_distances = face_recognition.face_distance(known_face_encodings, uf)
      best_match_index = np.argmin(face_distances)
      
      if matches[best_match_index]:
        name = known_face_names[best_match_index]
        
      if matches:
        results.append("{dimensions:("+str(left) + "," + str(top) + "," + str(right) + "," + str(bottom) + "), name:'" + name + "'}")
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
      return self.detect_faces_in_image(file)

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000)
