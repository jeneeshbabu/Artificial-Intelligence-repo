#Import Libraries
import cv2
import numpy as np

#Initialise Weights
net = cv2.dnn.readNet("yolov3_training.weights", "yolov3_testing.cfg")
classes = ["Weapon"]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def obj_detection(img):

    height, width, channels = img.shape
    
    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    #Draw and Label Bounding Box    
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 255)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 2, color, 2)
            
def detect_img(val):
    
    #Loading image
    img = cv2.imread(val)
    #img = cv2.resize(img, None, fx=0.4, fy=0.4)
    obj_detection(img)
    cv2.imshow("Image", img)
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()
        
def detect_vid(val):

    # for video capture
    cap = cv2.VideoCapture(val)
    while True:
        _, img = cap.read()
        height, width, channels = img.shape
        obj_detection(img)
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

def value():
    
    val = input("Enter Filename: ")

    if('.mp4' in val or 'mkv' in val or 'mov' in val):
        detect_vid(val)
    elif('.jpg' in val or '.png' in val or 'jpeg' in val):
        detect_img(val)
    elif val == '':
        detect_vid(0)
    else:
        print("Please input in Correct Format")
        value()

value()    


