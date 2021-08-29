import paho.mqtt.client as mqtt
import cv2 as cv
import numpy as np
import argparse
from detecto.core import Model

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

SERVER_PORT = 1883
TOPIC = 'ObjectDetection-tx'
FILENAME = 'model_weights.pth'

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--ip", help="mqtt server IP address", default="localhost")
parser.add_argument("-u", "--username", help="user name", default="simulator")
parser.add_argument("-p", "--password", help="password", default="simulator")
args = parser.parse_args()

labels = ['outlet', 'mouth', 'earth terminal']
saved_model = Model.load(FILENAME, labels)

def on_connect(client, userdata, flags, rc):
    print("Connected")
    client.subscribe(TOPIC)

def on_message(client, userdata, msg):
    array = np.frombuffer(msg.payload, dtype=np.uint8)
    img = cv.imdecode(array, cv.IMREAD_COLOR)
    
    # Detecto (based on pytorch)
    results = saved_model.predict(img)
    for i in range(len(results[0])):
        label = results[0][i]
        rect = results[1][i]
        acc = results[2][i]
        cv.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), 1)

    cv.imshow("ObjectDetection", img)
    cv.waitKey(1)

if __name__ == "__main__":

    client = mqtt.Client(client_id="OpenCvClient")
    if args.username and args.password:
        client.username_pw_set(args.username, args.password)
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(args.ip, SERVER_PORT, keepalive=60, bind_address="")

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        quit()
