import os
import pickle
import cv2

from flask import Flask, render_template, request
from random import random
from m_yolov6 import yolov6

# Load model once
yolo_v6_model = yolov6("weights/yolov6n.pt","cpu","data/coco.yaml", 640, True)

# Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static"

@app.route("/", methods=['GET', 'POST'])
def home_page():
    if (request.method == "POST"):
        # Upload picture
        try:
            image = request.files['file']            
            if (image):
                # Save uploaded img to ['UPLOAD_FOLDER']
                upload_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
                image.save(upload_path)

                # Loads img with OpenCV
                frame = cv2.imread(upload_path)

                # Inference
                frame, num_detected = yolo_v6_model.infer(frame, conf_thres=0.6, iou_thres=0.45)

                if (num_detected != 0):
                    # Overwrite the img with the one with bounding boxes placed
                    cv2.imwrite(upload_path, frame)

                    # Return result img to render
                    return render_template("index.html", result_image = image.filename, msg="Tải file lên thành công", num_detected = num_detected)
                else:
                    return render_template('index.html', result_image = image.filename, msg='Không nhận diện được vật thể', num_detected = num_detected)
            else:
                return render_template('index.html', msg='Hãy chọn file ảnh tải lên')

        except Exception as ex:
            print(ex)
            return render_template('index.html', msg='Không nhận diện được vật thể')

    else:
        # Load page
        return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)