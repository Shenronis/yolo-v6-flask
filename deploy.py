import os
import pickle
import cv2

from flask import Flask, render_template, request
from random import random
from m_yolov6 import my_yolov6

# Load model once
yolov6_model = my_yolov6("weights/yolov6s.pt","cpu","data/coco.yaml", 640, True)

# Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static"

@app.route("/", methods=['GET', 'POST'])
def home_page():
    # Upload picture
    if request.method == "POST":
        try:
            image = request.files['file']            
            if image:
                # Save uploaded img to ['UPLOAD_FOLDER']
                path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
                image.save(path_to_save)

                # Loads img with OpenCV
                frame = cv2.imread(path_to_save)

                # Inference
                frame, ndet = yolov6_model.infer(frame, conf_thres=0.6, iou_thres=0.45)

                if ndet!=0:
                    # Overwrite the img with the one with bounding boxes placed
                    cv2.imwrite(path_to_save, frame)

                    # Return result img to render
                    return render_template("index.html", result_image = image.filename, msg="Tải file lên thành công", ndet = ndet)
                else:
                    return render_template('index.html', result_image = image.filename, msg='Không nhận diện được vật thể', ndet = ndet)
            else:
                return render_template('index.html', msg='Hãy chọn file ảnh tải lên')

        except Exception as ex:
            print(ex)
            return render_template('index.html', msg='Không nhận diện được vật thể')

    # Load page
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)