import os
import cv2
import sys
from zipfile import ZipFile
from urllib.request import urlretrieve

def download_and_unzip(url, save_path:str):
    print(f"Downloading and extracting assets...", end="")

    urlretrieve(url, save_path)

    try:
        with ZipFile(save_path) as z:
            folder = save_path.split('.')[0]
            z.extractall(folder)
            os.remove(save_path)

        print("Done")
    
    except Exception as e:
        print("|nInvalid file.", e)

URL = r"https://www.dropbox.com/s/efitgt363ada95a/opencv_bootcamp_assets_12.zip?dl=1"

assets_zip_path = os.path.join("opencv_bootcamp_assets_12.zip")
folder_path = assets_zip_path.split('.')[0]

if not os.path.exists(folder_path):
    download_and_unzip(URL, assets_zip_path)


# ============================================================================================ #
s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

source = cv2.VideoCapture(s)

win_name = "Camera Preview"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

net = cv2.dnn.readNetFromCaffe ("opencv_bootcamp_assets_12/deploy.prototxt", "opencv_bootcamp_assets_12/res10_300x300_ssd_iter_140000_fp16.caffemodel")
# model parameters
in_width = 300
in_height = 300
mean = [104, 117, 123]
conf_threshold = 0.7

while cv2.waitKey(1) != 27:
    has_frame, frame = source.read()
    if not has_frame:
        break
    frame = cv2.flip(frame, 1)
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (in_width, in_height), mean, swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward()
    print(f"Detections: {detections}")

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x_left_bottom = int(detections[0, 0, i, 3] * frame_width)
            y_left_bottom = int(detections[0, 0, i, 4] * frame_height)
            x_right_top = int(detections[0, 0, i, 5] * frame_width)
            y_right_top = int(detections[0, 0, i, 6] * frame_height)

            cv2.rectangle(frame, (x_left_bottom, y_left_bottom), (x_right_top, y_right_top), (0, 255, 0))
            label = "Confidence: %.4f" % confidence
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            cv2.rectangle(
                frame,
                (x_left_bottom, y_left_bottom - label_size[1]),
                (x_left_bottom + label_size[0], y_left_bottom + base_line),
                (255, 255, 255)
            )

            cv2.putText(frame, label, (x_left_bottom, y_left_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    t, _ = net.getPerfProfile()
    label = "Inference time: %.2f ms" % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    cv2.imshow(win_name, frame)

source.release()
cv2.destroyWindow(win_name)
        

