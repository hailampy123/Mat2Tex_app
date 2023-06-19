from flask import Flask, Response, request, jsonify
from flask_cors import CORS

import os
import ast
# import numpy as np
# import cv2
# from PIL import Image
import base64

# Google cloud packages
from typing import Dict, List, Union
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
from google.protobuf.json_format import MessageToDict

app = Flask(__name__)
CORS(app)

# # Upload image from web interface
# path = os.getcwd()
# UPLOAD_FOLDER = os.path.join(path, 'static\\')
# if not os.path.isdir(UPLOAD_FOLDER):
#     os.mkdir(UPLOAD_FOLDER)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# GCP materials
project="intrepid-charge-382512"
detection_endpoint_id="1243802737712300032"
extraction_endpoint_id="3611007291848916992"
location="asia-southeast2"
api_endpoint = "asia-southeast2-aiplatform.googleapis.com"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getcwd() + "/vertex-ai-user-service-account.json"

def predict_custom_trained_model_sample(
    project: str,
    endpoint_id: str,
    instances: Union[Dict, List[Dict]],
    location: str,
    api_endpoint: str,):
    """
    `instances` can be either single instance of type dict or a list
    of instances.
    """
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    # The format of each instance should conform to the deployed model's prediction input schema.
    # instances = instances if type(instances) == list else [instances]
    # instances = [
    #     json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
    # ]
    # print(instances)
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    # The predictions are a google.protobuf.Value representation of the model's predictions.
    return response

# ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/isalive")
def is_alive():
    print("/isalive request")
    status_code = Response(status=200)
    return status_code

@app.route('/detect', methods = ['POST'])
def detect():
    file = request.files['file']
    # print(app.config['UPLOAD_FOLDER'])
    # file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    # predicting_img_path = UPLOAD_FOLDER + file.filename if file.filename else UPLOAD_FOLDER
    
    # create data input to load to deployed model
    # origin_img = Image.open(file)
    # img_arr = np.array(origin_img)
    # _, im_arr = cv2.imencode('.jpg', img_arr)
    # im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(file.read())
    im_b64 = im_b64.decode('utf-8') # Convert to string type to send to flask backend

    instance = {'im_base64': im_b64}
    
    response = predict_custom_trained_model_sample(
        project = project,
        endpoint_id = detection_endpoint_id,
        location = location,
        instances = [instance],
        api_endpoint = api_endpoint
    )
    response_json = MessageToDict(response._pb)
    detected_obj = response_json['predictions']
    
    return jsonify({
        "predictions": detected_obj
    })

@app.route('/extract', methods = ['POST'])
def extract():
    file = request.files['file']
    # point_list = request.files['point_list']
    req_json = request.form.get('point_list')
    print(req_json)
    point_list = ast.literal_eval(req_json)
    print(len(point_list))
    # file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
    # predicting_img_path = UPLOAD_FOLDER + file.filename if file.filename else UPLOAD_FOLDER
    
    # create data input to load to deployed model
    # origin_img = Image.open(file)
    # img_arr = np.array(origin_img)
    # _, im_arr = cv2.imencode('.jpg', img_arr)
    im_b64 = base64.b64encode(file.read())
    im_b64 = im_b64.decode('utf-8') # Convert to string type to send to flask backend

    instance = {'point_list': point_list, 'im_base64': im_b64}
    
    response = predict_custom_trained_model_sample(
        project = project,
        endpoint_id = extraction_endpoint_id,
        location = location,
        instances = [instance],
        api_endpoint = api_endpoint
    )
    response_json = MessageToDict(response._pb)
    results = response_json['predictions']
    
    return jsonify({
        "predictions": results
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=True, host="0.0.0.0", port=port)
