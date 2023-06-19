from flask import Flask, Response, request, jsonify
from flask_cors import CORS

from ultralytics import YOLO
from PIL import Image

import base64
from io import BytesIO

app = Flask(__name__)
CORS(app)

@app.route("/isalive")
def is_alive():
    print("/isalive request")
    status_code = Response(status=200)
    return status_code

@app.route('/predict', methods = ['POST'])
def upload_page():
    print("/predict request")
    
    req_json = request.get_json()
    json_instances = req_json['instances']
    
    im_base64 = json_instances[0]['im_base64']
    im_b64 = im_base64
    im_b64 = im_b64.encode('utf-8')
    im_bytes = base64.b64decode(im_b64)   # im_bytes is a binary image
    predicting_img = BytesIO(im_bytes)  # convert image to file-like object
    
    detected_model = YOLO('best.pt')  # load a pretrained model (recommended for training)
    image = Image.open(predicting_img)
    results = detected_model.predict(image) # Use the model
    
    classes = ['embedded', 'isolated']

    pred = results[0].boxes # Get bbox coordinate and class info
    pointList = pred[pred.conf > 0.6].boxes.tolist()

    for i in pointList:
        i[-1] = classes[int(i[-1])]
    print(pointList)
    
    return jsonify({
        "predictions": pointList
    })
    
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)