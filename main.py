import cv2
import random
import time


def getFaceBox(faceNet, frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    faceBoxes = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return frame, faceBoxes


faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

video = cv2.VideoCapture(0)

padding = 20
ghost_active = False
ghost_position = None
start_time = time.time()  # Timer to control ghost appearance
ghost_start_time = None

while True:
    hasFrame, vidFrame = video.read()

    if not hasFrame:
        cv2.waitKey()
        break

    frame, faceBoxes = getFaceBox(faceNet, vidFrame)

    # Detect real faces
    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - padding):min(faceBox[3] + padding, frame.shape[0] - 1),
               max(0, faceBox[0] - padding):min(faceBox[2] + padding, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        gender = genderList[genderPred[0].argmax()]

        ageNet.setInput(blob)
        agePred = ageNet.forward()
        age = ageList[agePred[0].argmax()]

        labelGender = "{}".format("Gender : " + gender)
        labelAge = "{}".format("Age : " + age + " Years")
        cv2.putText(frame, labelGender, (faceBox[0], faceBox[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, labelAge, (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA)

    # Trigger ghost detection after 8 seconds
    current_time = time.time()

    elapsed_time = current_time - start_time

    # Ghost appears after 10 seconds
    if elapsed_time > 1000000 and not ghost_active:
        ghost_active = True
        ghost_start_time = current_time

        # If no ghost position yet, place it at a random face-like location
        ghost_position = random.choice(faceBoxes) if faceBoxes else [200, 200, 300, 300]

    # Display ghost for 2 minutes
    if ghost_active and current_time - ghost_start_time < 120:
        x1, y1, x2, y2 = 50, 40, 140, 160
        overlay = frame.copy()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        alpha = 0.5  # Transparency
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        labelGhost = "Gender : Unknown"
        labelGhostAge = "Age : 496-635 Years"
        cv2.putText(frame, labelGhost, (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 0, 225), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, labelGhostAge, (x1 + 10, y1 + 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (0, 0, 225), 2,
                    cv2.LINE_AA)
    elif ghost_active:
        # Deactivate ghost after 2 minutes
        ghost_active = False
        ghost_start_time = None

    cv2.imshow("Age-Gender Detector with Ghost", frame)
    if cv2.waitKey(10) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()