import face_recognition
import cv2
import os

LABELED_IMAGES_DIR = "./resources/labeled_images"
TEST_IMAGES_DIR = "./resources/test_images"

TOLERANCE = 0.6
MODEL = 'cnn'

FRAME_THICKNESS = 3
FONT_THICKNESS = 2

print("Processing labeled images")

labeledImages = []
labels = []

#loop through all the directories within labeled_images
for name in os.listdir(LABELED_IMAGES_DIR):
    #loop through all the files within
    for filename in os.listdir(f"{LABELED_IMAGES_DIR}/{name}"):

        image = face_recognition.load_image_file(f"{LABELED_IMAGES_DIR}/{name}/{filename}")
        encoding = face_recognition.face_encodings(image)[0]

        labeledImages.append(encoding)
        labels.append(name)

print("processing test images")
for filename in os.listdir(TEST_IMAGES_DIR):
    print(filename)
    image = face_recognition.load_image_file(f"{TEST_IMAGES_DIR}/{filename}")
    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for faceEncoding, faceLocation in zip(encodings, locations):
        results = face_recognition.compare_faces(labeledImages, faceEncoding, TOLERANCE)
        match = None
        print(f"Face locations: {faceLocation}")
        if True in results:
            match = labels[results.index(True)]
            print(f"Match found: {match}")

            #Draw rectangle around face
            topLeft = (faceLocation[3], face_recognition[0])
            bottomRight = (faceLocation[1], face_recognition[2])
            color = [ 0, 255, 0]
            cv2.rectangle(image, topLeft, bottomRight, color, FRAME_THICKNESS)

            #Draw label
            topLeft = (faceLocation[3], face_recognition[2])
            bottomRight = (faceLocation[1], face_recognition[2]+22)
            cv2.rectangle(image, topLeft, bottomRight, color, cv2.FILLED)
            cv2.putText(image, match, (faceLocation[3]+10, face_recognition[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)

    cv2.imShow(filename, image)
    cv2.waitKey(0)
    cv2.destroyWindow(filename)