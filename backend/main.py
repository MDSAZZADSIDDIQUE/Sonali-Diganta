from flask_socketio import SocketIO, emit
from flask import Response
import numpy as np
from ultralytics import YOLO
import cvzone
import cv2
import random
from chat import get_response
from flask import Flask, jsonify
import tensorflow as tf
from keras.preprocessing import image
from flask_cors import CORS, cross_origin
from flask import request
import math
import pickle

app = Flask(__name__)

CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
class_name = [
    'Apple Apple scab',
    'Apple Black rot',
    'Apple Cedar apple rust',
    'Apple healthy',
    'Blueberry healthy',
    'Cherry (including sour) healthy',
    'Cherry (including sour) Powdery mildew',
    'Corn (maize) Cercospora leaf spot Gray leaf spot',
    'Corn (maize) Common rust',
    'Corn (maize) healthy',
    'Corn (maize) Northern Leaf Blight',
    'Grape Black rot',
    'Grape Esca (Black Measles)',
    'Grape healthy',
    'Grape Leaf blight (Isariopsis Leaf Spot)',
    'Orange Haunglongbing (Citrus greening)',
    'Peach Bacterial spot',
    'Peach healthy',
    'Pepper, bell Bacterial spot',
    'Pepper, bell healthy',
    'Potato Early blight',
    'Potato healthy',
    'Potato Late blight',
    'Raspberry healthy',
    'Soybean healthy',
    'Squash Powdery mildew',
    'Strawberry healthy',
    'Strawberry Leaf scorch',
    'Tomato Bacterial spot',
    'Tomato Early blight',
    'Tomato healthy',
    'Tomato Late blight',
    'Tomato Leaf Mold',
    'Tomato Septoria leaf spot',
    'Tomato Spider mites Two spotted spider mite',
    'Tomato Target Spot',
    'Tomato Tomato mosaic virus',
    'Tomato Tomato Yellow Leaf Curl Virus'
]


@app.route('/plant-disease-detection', methods=['POST'])
def plantDiseaseDetection():
    model = tf.keras.models.load_model("plant_disease_detection.h5")
    imagefile = request.files['image']
    filepath = "tmp/temp.jpg"
    imagefile.save(filepath)
    i = image.load_img(filepath, target_size=(224, 224))
    i = image.img_to_array(i)
    i = i.reshape(1, 224, 224, 3)
    p = model.predict(i)
    oi = class_name[p.argmax()]
    score = math.floor(float(p[0].max()) * 100)
    return jsonify({'class': oi, 'confidence': score})


plant_growth_stages = ['Flowering', 'Fruit Development', 'Germination',
                       'Pollination', 'RipeningMaturation', 'Seedling/Establishment', 'Vegetative Growth']


@app.route('/plantgrowthstage', methods=['POST'])
def plantGrowthStage():
    model = tf.keras.models.load_model("plant_growth_stage.h5")
    imagefile = request.files['image']
    filepath = "tmp/temp.jpg"
    imagefile.save(filepath)
    i = image.load_img(filepath, target_size=(64, 64))
    i = image.img_to_array(i)
    i = i.reshape(1, 64, 64, 3)
    p = model.predict(i)
    oi = plant_growth_stages[p.argmax()]
    score = math.floor(float(p[0].max()) * 100)
    return jsonify({'class': oi, 'confidence': score})


@app.route('/croprecommendation', methods=['POST'])
def cropRecommendation():
    with open('NaiveBayes.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    prediction = loaded_model.predict(
        [[float(request.form['n']), float(request.form['p']), float(request.form['k']), float(request.form['temperature']), float(request.form['humidity']), float(request.form['ph']), float(request.form['rainfall'])]])

    return jsonify({'class': str(prediction[0])})


@app.route('/cropyieldprediction', methods=['POST'])
def cropYieldPrediction():
    with open('RF.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    prediction = loaded_model.predict(
        [[float(request.form['Soil pH']), float(request.form['Temperature']), float(request.form['Precipitation']), float(request.form['Sunlight Exposure']), float(request.form['Fertilizer Usage'])]])

    return jsonify({'class': str(prediction[0])})


fertilizer_names = ['Urea', 'DAP', '14-35-14', '28-28', '17-17-17', '20-20',
                    '10-26-26']


@app.route('/fertilizerrecommendation', methods=['POST'])
def fertilizerRecommendation():
    with open('svm_pipeline.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    prediction = loaded_model.predict(
        [[float(request.form['Temparature']), float(request.form['Humidity']), float(request.form['Moisture']), random.randint(1, 11), random.randint(1, 5), float(request.form['Nitrogen']), float(request.form['Potassium']), float(request.form['Phosphorous'])]])

    return jsonify({'class': fertilizer_names[prediction[0]]})


@app.route("/predictchat", methods=["POST"])
def predictChat():
    text = request.get_json().get("message")
    response = get_response(text)
    message = {"answer": response}
    response = jsonify(message)

    return response


pest = ['aphids', 'armyworm', 'beetle', 'bollworm',
        'grasshopper', 'mites', 'mosquito', 'sawfly', 'stem_borer']


@app.route('/pest-detection', methods=['POST'])
def pestDetection():
    model = tf.keras.models.load_model("pest_detection.h5")
    imagefile = request.files['image']
    filepath = "tmp/temp.jpg"
    imagefile.save(filepath)
    i = image.load_img(filepath, target_size=(224, 224))
    i = image.img_to_array(i)
    i = i.reshape(1, 224, 224, 3)
    p = model.predict(i)
    oi = pest[p.argmax()]
    score = math.floor(float(p[0].max()) * 100)
    return jsonify({'class': oi, 'confidence': score})


@app.route("/")
@cross_origin()
def hello_world():
    return "<p>Hello, World!</p>"


classNames = ['light']
model = YOLO("best.pt")
cap = cv2.VideoCapture("Field Data.mp4")


def hello():
    zone1 = 0
    zone2 = 0
    zone3 = 0
    zone4 = 0

    limits = [500, 300, 1000, 300]
    limits2 = [0, 300, 500, 300]
    limits3 = [500, 150, 1000, 150]
    limits4 = [0, 150, 500, 150]

    while True:
        success, img = cap.read()
        results = model(img, stream=True)
        detections = np.empty((0, 5))
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
                w, h = x2 - x1, y2 - y1
                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])
                if conf > 0.5:
                    cvzone.cornerRect(img, (x1, y1, w, h))
                    cx, cy = x1 + w // 2, y1 + h // 2
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                    cv2.putText(img, str(
                        f"{cx} & {cy}"), (155, 300), cv2.FONT_HERSHEY_PLAIN, 3, (50, 50, 255), 3)

                    if limits[0] < cx < limits[2] and limits[1] - 150 < cy < limits[1] + 150:
                        zone1 = zone1 + 1
    #                     cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
                    if limits2[0] < cx < limits2[2] and limits2[1] - 150 < cy < limits2[1] + 150:
                        zone2 = zone2 + 1
    #                     cv2.line(img, (limits2[0], limits2[1]), (limits2[2], limits2[3]), (0, 255, 0), 5)
                    if limits3[0] < cx < limits3[2] and limits3[1] - 150 < cy < limits3[1] + 150:
                        zone3 = zone3 + 1
    #                     cv2.line(img, (limits3[0], limits3[1]), (limits3[2], limits3[3]), (0, 255, 0), 5)
                    if limits4[0] < cx < limits4[2] and limits4[1] - 150 < cy < limits4[1] + 150:
                        zone4 = zone4 + 1
    #                     cv2.line(img, (limits4[0], limits4[1]), (limits4[2], limits4[3]), (0, 255, 0), 5)

    #     cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    #     cv2.line(img, (limits2[0], limits2[1]), (limits2[2], limits2[3]), (255, 0, 0), 5)
    #     cv2.line(img, (limits3[0], limits3[1]), (limits3[2], limits3[3]), (0, 255, 0), 5)
    #     cv2.line(img, (limits4[0], limits4[1]), (limits4[2], limits4[3]), (255, 0, 255), 5)

        cv2.putText(img, str(
            f"Zone 1: {zone4}s"), (155, 100), cv2.FONT_HERSHEY_PLAIN, 3, (50, 50, 255), 3)
        cv2.putText(img, str(
            f"Zone 2: {zone3}s"), (555, 100), cv2.FONT_HERSHEY_PLAIN, 3, (50, 50, 255), 3)
        cv2.putText(img, str(
            f"Zone 3: {zone2}s"), (155, 400), cv2.FONT_HERSHEY_PLAIN, 3, (50, 50, 255), 3)
        cv2.putText(img, str(
            f"Zone 4: {zone1}s"), (555, 400), cv2.FONT_HERSHEY_PLAIN, 3, (50, 50, 255), 3)

        cv2.imshow("Sunlight amount detector", img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


socketio = SocketIO(app)


@socketio.on('connect', namespace='/video')
def test_connect():
    """Video streaming function."""
    cap = cv2.VideoCapture("Field Data.mp4")
    while True:
        success, img = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            emit('frame', {'image': frame})


# socketio.run(app)
