from flask import Flask, Response, request, render_template, jsonify
from flask_cors import CORS
import extracting


import json
import re
import torch
from torch import nn, Tensor
from abc import ABC, abstractmethod

import base64
from io import BytesIO


class Text(ABC):
    def __init__(self):
        self.pad_id = 0
        self.sos_id = 1
        self.eos_id = 2

    @abstractmethod
    def tokenize(self, formula: str):
        pass

    def int2text(self, x: Tensor):
        return " ".join([self.id2word[i] for i in x if i > self.eos_id])

    def text2int(self, formula: str):
        return torch.LongTensor([self.word2id[i] for i in self.tokenize(formula)])

class Text100k(Text):
    def __init__(self):
        super().__init__()
        self.id2word = json.load(open("100k_vocab.json", "r"))
        self.word2id = dict(zip(self.id2word, range(len(self.id2word))))
        self.TOKENIZE_PATTERN = re.compile(
            "(\\\\[a-zA-Z]+)|" + '((\\\\)*[$-/:-?{-~!"^_`\[\]])|' + "(\w)|" + "(\\\\)"
        )
        self.n_class = len(self.id2word)

    def tokenize(self, formula: str):
        tokens = re.finditer(self.TOKENIZE_PATTERN, formula)
        tokens = list(map(lambda x: x.group(0), tokens))
        tokens = [x for x in tokens if x is not None and x != ""]
        return tokens


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
    
    point_list = json_instances[0]['point_list']
    
    im_base64 = json_instances[0]['im_base64']
    
    # file = request.files['file']
    # # point_list = request.files['point_list']
    # req_json = request.form.get('point_list')
    # print(req_json)
    # point_list = ast.literal_eval(req_json)
    # print(len(point_list))
    
    im_b64 = im_base64
    im_b64 = im_b64.encode('utf-8')
    im_bytes = base64.b64decode(im_b64)   # im_bytes is a binary image
    predicting_img = BytesIO(im_bytes)  # convert image to file-like object
    
    result = extracting.predicting(point_list, predicting_img, 'model_best_checkpoint.pth.tar', 'transfer_model.ckpt')

    return jsonify({
        "predictions": result
    })

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)
