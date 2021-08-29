import cv2 as cv
import argparse
from detecto.core import Model

parser = argparse.ArgumentParser()
parser.add_argument("file", help="file name", type=str)
args = parser.parse_args()

FILENAME = 'model_weights.pth'

labels = ['outlet', 'mouth', 'earth terminal']
saved_model = Model.load(FILENAME, labels)

def infer(img, threshold):
    results = saved_model.predict(img)
    inference_results = []
    for i in range(len(results[0])):
        label = results[0][i]
        rect = results[1][i]
        acc = results[2][i]
        inference_results.append((label, rect, acc))
        if acc > threshold:
            cv.rectangle(mat, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), 1)

if __name__ == "__main__":
    mat = cv.imread(args.file)
    infer(mat, 0.8)
    cv.imshow("test", mat)
    cv.waitKey(0)
    cv.destroyAllWindows()

