from flask import Flask, render_template, request, redirect, url_for, jsonify
from imageio import imread
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import cv2
import base64
import io

# UPLOAD_FOLDER = './temp'
ALLOWED_EXT = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# combined = load_model('model/combined.h5')
# combined = load_model('model/combined_new.h5')
general = load_model('../model/check_letter.h5')
specific = load_model('../model/digit_symb.h5')
digits = load_model('../model/digits.h5')
letters = load_model('../model/letters.h5')
symbols = load_model('../model/symbols.h5')
dict_letters = {k: v for k,
                v in enumerate([chr(i+65)+"/"+chr(i+97) for i in range(26)])}

dict_symbols = {
    0: ',',
    1: '!',
    2: '.',
    3: '?'
}


def recrop_transform(img):
    img_gr = img
    padding = 0.25
    padding_idx = 0
    min_idx = list(map(np.min, np.nonzero(img_gr)))
    max_idx = list(map(np.max, np.nonzero(img_gr)))
    y_diff = max_idx[0] - min_idx[0]
    x_diff = max_idx[1] - min_idx[1]

    check = y_diff < x_diff

    if not check:
        padding_idx = int(0.25 * y_diff)
        min_idx[1] = max(0, min_idx[1] - (y_diff - x_diff)//2)
        max_idx[1] = min_idx[1] + y_diff
    else:
        padding_idx = int(0.25 * x_diff)
        min_idx[0] = max(0, min_idx[0] - (x_diff - y_diff)//2)
        max_idx[0] = min_idx[0] + x_diff

    min_idx[0] = max(0, min_idx[0] - padding_idx)
    min_idx[1] = max(0, min_idx[1] - padding_idx)
    max_idx[0] = min(img.shape[0], max_idx[0] + padding_idx)
    max_idx[1] = min(img.shape[1], max_idx[1] + padding_idx)

    np_cropped = np.array(img)[min_idx[0]:max_idx[0], min_idx[1]:max_idx[1]]
    return np_cropped


@app.route('/')
def main():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = str.encode(request.form['image'].replace(
            'data:image/jpeg;base64,', ''))
        if file:
            nparr = imread(io.BytesIO(base64.b64decode(file)))
            img_np = cv2.cvtColor(nparr, cv2.COLOR_RGB2GRAY)
            img_np = cv2.bitwise_not(img_np)
            img_np = recrop_transform(img_np)
            img_np = cv2.resize(img_np, (28, 28))
            img_np = img_np/255.0
            # plt.imshow(img_np, cmap='gray')
            # plt.show()

            img_np = img_np[:, :, np.newaxis]
            result = general.predict(np.array([img_np]))
            print(result)
            result = np.round(result[0])

            if result[0] == 0:
                result = letters.predict(np.array([img_np]))
                result = np.argmax(result[0])
                result = dict_letters[result]
            elif result[0] == 1:
                result = specific.predict(np.array([img_np]))
                result = np.round(result[0])
                if result[0] == 0:
                    result = digits.predict(np.array([img_np]))
                    result = np.argmax(result[0])
                elif result[0] == 1:
                    result = symbols.predict(np.array([img_np]))
                    result = np.argmax(result[0])
                    result = dict_symbols[result]

            # image = base64.b64decode(file)
            # file_handle = open('./temp/image.jpg', 'wb')
            # file_handle.write(image)
            # file_handle.close()

            return jsonify({'status': 'success', 'result': str(result)})
        return jsonify({'status': 'error'})
