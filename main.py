import cv2
import numpy as np
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import datetime as dt
import pandas as pd
import paho.mqtt.client as mqtt
import time
import base64
import os
import mediapipe as mp
import warnings
import wiringpi
from wiringpi import GPIO

mp_solutions = mp.solutions

# Load MobileFaceNet frozen graph
def load_frozen_graph(pb_file_path):
    with tf.io.gfile.GFile(pb_file_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.compat.v1.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph

graph = load_frozen_graph('/root/Documents/face_recognition/New Algo/MobileFaceNet_9925_9680.pb')
input_tensor = graph.get_tensor_by_name("input:0")
output_tensor = graph.get_tensor_by_name("embeddings:0")

sess = tf.compat.v1.Session(graph=graph)

face_detection = mp_solutions.face_detection.FaceDetection(model_selection= 0, min_detection_confidence=0.9)

# pre-trained Random Forest classifier
rf_classifier = joblib.load('/root/Documents/face_recognition/Final/svm_final.joblib')

mp_draw = mp.solutions.drawing_utils

# Preprocessing function
def preprocess_image(image):
    if image.size == 0:
        raise ValueError("Empty image provided for preprocessing.")
    image = cv2.resize(image, (112, 112))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# Get face embedding function
def get_face_embedding(face_image):
    preprocessed_image = preprocess_image(face_image)
    embedding = sess.run(output_tensor, feed_dict={input_tensor: preprocessed_image})
    return embedding.flatten()

# Recognize face function with confidence threshold
def recognize_face(face_image):
    embedding = get_face_embedding(face_image)
    probabilities = rf_classifier.predict_proba([embedding])[0]
    max_index = np.argmax(probabilities)
    max_probability = probabilities[max_index]
    return max_index ,max_probability


def label_names(class_in):
  try:
    filename = '/root/Documents/face_recognition/Final/labels.txt'
    with open(filename, 'r') as file:
      lines = file.readlines()
      if 0 <= class_in < len(lines):
          return lines[class_in].strip()
      else:
          return None
  except FileNotFoundError:
    print("File not found.")
    return None

class FaceRecognition:
    def __init__(self, broker_address="localhost"):
        self.attendance_records = []
        self.predicted_names = []
        self.broker_address = broker_address
        self.message_topic = "hello/world"
        self.image_topic = "image/upload"
        self.msg_client = mqtt.Client()
        self.img_client = mqtt.Client()

        # GPIO setup
        wiringpi.wiringPiSetup()
        wiringpi.pinMode(2, GPIO.OUTPUT)
        wiringpi.pinMode(4, GPIO.INPUT)
        
    def recognize_faces(self, frame):
        current_time = dt.datetime.now().strftime('%I:%M:%S %p')
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                (x, y, w, h) = (int(bboxC.xmin * iw), int(bboxC.ymin * ih), 
                                int(bboxC.width * iw), int(bboxC.height * ih))
                
                if x < 0 or y < 0 or x + w > iw or y + h > ih:
                    continue  # Skip invalid bounding boxes
                
                face_image = frame[y:y+h, x:x+w]
                
                if face_image.size == 0:
                    continue  # Skip empty face images
                
                prediction, probability = recognize_face(face_image)
                
                # confidence thresold
                if prediction is not None and probability >= 0.50:
                    predicted_name = label_names(int(prediction))

                    current_date = dt.datetime.now().strftime('%d-%m-%y')
                    if not any(record['Name'] == predicted_name for record in self.attendance_records):
                        self.attendance_records.append({'Name': predicted_name, 'Date': current_date, 'Time': current_time})
                    cv2.putText(frame, predicted_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    self.predicted_names = [predicted_name]
                    break  # Break after processing the first face with high confidence
                else:
                    cv2.putText(frame, "Unknown Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    #cv2.putText(frame, "Unknown Person", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0>
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    self.predicted_names = ["Unknown Person"]
                    
                
            attendance_df = pd.DataFrame(self.attendance_records)
            attendance_df.to_csv('attendance_records.csv', index=False)

    def run(self):
        button_status = wiringpi.digitalRead(4)
        cap = cv2.VideoCapture(1)
        self.msg_client.connect(self.broker_address)
        self.img_client.connect(self.broker_address, 1883, 60)
        wiringpi.digitalWrite(2, GPIO.HIGH)  # uncomment to operate door lock (puts in closed state)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
        
            time.sleep(0.01)
            key = cv2.waitKey(1) & 0xFF
            self.recognize_faces(frame)
            try:
                if not self.predicted_names or "Unknown Person" in self.predicted_names:
                    # wiringpi.digitalWrite(2, GPIO.HIGH)  # uncomment to operate door lock (puts in closed state)
                    message = "Person unrecognised, Entry denied"
                    self.msg_client.publish(self.message_topic, message)
                    filename = f"webcam_capture_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    with open(filename, "rb") as image_file:
                        image_data = base64.b64encode(image_file.read())
                        self.img_client.publish(self.image_topic, image_data)
                        os.remove(filename)

                    perm = input("Press 'a' to allow person")
                    if perm == "a":
                        print("person allowed !")
                        wiringpi.digitalWrite(2, GPIO.LOW)  # uncomment to operate door lock (puts in closed state)
                        time.sleep(2)
                        wiringpi.digitalWrite(2, GPIO.HIGH)  # uncomment to operate door lock (puts in open state)
                else:
                    predicted_name = self.predicted_names[0]
                    message = f"Welcome {predicted_name}! to Tlc Polymers Ltd."
                    self.msg_client.publish(self.message_topic, message)
                    wiringpi.digitalWrite(2, GPIO.LOW)  # uncomment to operate door lock (puts in open state)
            except KeyboardInterrupt:
                print("Exiting")

            cv2.imshow('Face Recognition', frame)
            if key == ord('q'):
                break
                
            # Uncomment to activate push button to click photos in case of False Negatives
            #if key == ord('p'):
            #if button_status == 0:
            #    filename = f"webcam_capture_{int(time.time())}.jpg"
            #    cv2.imwrite(filename, frame)
            #    with open(filename, "rb") as image_file:
            #        image_data = base64.b64encode(image_file.read())
            #        self.img_client.publish(self.image_topic, image_data)
            #        os.remove(filename)

        cap.release()
        self.msg_client.disconnect()
        self.img_client.disconnect()
        cv2.destroyAllWindows()
    
face_recognition = FaceRecognition()
face_recognition.run()
