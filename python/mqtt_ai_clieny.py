import paho.mqtt.client as mqtt
import cv2 as cv
import numpy as np
import argparse
from detecto.core import Model
import time

SERVER_PORT = 1883
TOPIC_TX = 'AI-tx'
TOPIC_RX = 'AI-rx'
FILENAME = 'model_weights.pth'

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--ip", help="mqtt server IP address", default="localhost")
parser.add_argument("-u", "--username", help="user name", default="simulator")
parser.add_argument("-p", "--password", help="password", default="simulator")
parser.add_argument("-t", "--threshold", help="accuracy threshold", type=float, default=0.8)
parser.add_argument("-r", "--resize", help="resize viewer image", type=float, default=1.0)
parser.add_argument("-d", "--devicename", help="device name", default="AI client")
args = parser.parse_args()

labels = ['outlet', 'mouth', 'earth terminal']
saved_model = Model.load(FILENAME, labels)

def on_connect(client, userdata, flags, rc):
    print("Connected")
    client.subscribe('{}/#'.format(TOPIC_TX))

def on_message(client, userdata, msg):
    sub_topic = msg.topic.split('/')[1:]
    command = sub_topic[0]
    params = []
    if len(sub_topic) > 1:
        params = sub_topic[1:]

    array = np.frombuffer(msg.payload, dtype=np.uint8)  # JPEG data
    src = cv.imdecode(array, cv.IMREAD_COLOR)  # JPEG to mat

    src = cv.resize(src, (int(src.shape[1]/args.resize), int(src.shape[0]/args.resize)))

    if command == 'ObjectDetection':  # Detecto (based on pytorch)
        results = saved_model.predict(src)
        inference_results = []
        for i in range(len(results[0])):
            label = results[0][i]
            rect = results[1][i]
            acc = results[2][i]
            inference_results.append((label, rect, acc))
            if acc > args.threshold:
                cv.rectangle(src, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), 1)
                cv.putText(src, "{} {:.2f}".format(label, acc), (rect[0], rect[1]-5), cv.FONT_HERSHEY_PLAIN, fontScale=1.0, color=(0, 0, 255), lineType=cv.LINE_AA)
        dst = src
    
    elif command == 'HistgramEqualization':
        grey = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        dst = cv.equalizeHist(grey)
    
    elif command == 'SobelFilter':
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        dst = cv.GaussianBlur(gray, (3, 3), 0)
        xy = params[0]
        if (xy == 'x'):
            dst = cv.Sobel(dst, cv.CV_8UC1, 1, 0, ksize=5)
        elif (xy == 'y'):
            dst = cv.Sobel(dst, cv.CV_8UC1, 0, 1, ksize=5)
        
        dst = cv.cvtColor(dst, cv.COLOR_BGR2RGB)

    elif command == 'MorphologicalTransformation':
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)        
        invert = params[0]
        kernel = np.ones((7, 7), np.uint8)
        dst = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel)
        if invert == 'true':
            dst = cv.bitwise_not(dst)        
        dst = cv.cvtColor(dst, cv.COLOR_BGR2RGB)

    binary_dst = cv.imencode('.jpg', dst)[1].tobytes()
    array = np.frombuffer(binary_dst, dtype=np.uint8)  # JPEG data
    dst = cv.imdecode(array, cv.IMREAD_COLOR)  # JPEG to mat

    cv.imshow("Viewer", dst)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":

    client = mqtt.Client(client_id=args.devicename)
    if args.username and args.password:
        client.username_pw_set(args.username, args.password)
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(args.ip, SERVER_PORT, keepalive=60, bind_address="")

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        quit()