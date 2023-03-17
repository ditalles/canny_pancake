from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import io
from PIL import Image
from selfie_line_drawing import face_detection, canny_edge, canny_edge_auto, laplacian_of_gaussian



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/line-drawing', methods=['POST'])
def line_drawing():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    method = request.form.get('method', 'canny')
    blur_kernel_size = int(request.form.get('blur_kernel_size', 5))
    low_threshold = int(request.form.get('low_threshold', 100))
    high_threshold = int(request.form.get('high_threshold', 200))

    input_image = Image.open(file.stream)
    input_image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)

    face_image = face_detection(input_image)

    if method == 'canny':
        edge_image = canny_edge(face_image, blur_kernel_size, low_threshold, high_threshold)
    elif method == 'auto_canny':
        edge_image = canny_edge_auto(face_image, blur_kernel_size)

    elif method == 'log':
        edge_image = laplacian_of_gaussian(face_image, blur_kernel_size)

    _, buffer = cv2.imencode('.jpg', edge_image)
    output_image_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'image': output_image_base64})

if __name__ == '__main__':
    app.run(debug=True, port=5005)
