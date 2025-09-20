import os
import cv2 as cv
import numpy as np

# face detection using haar cascade

# img=cv.imread(r"C:\Users\balam\OneDrive\Pictures\Whatsapp images\IMG-20250116-WA0259.jpg")
# gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
# haar_cascade=cv.CascadeClassifier(cv.data.haarcascades + "haar_face.xml")

# faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

# print(f"Number of faces found = {len(faces_rect)}")

# for (x, y, w, h) in faces_rect:
    # cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# cv.imshow("Detected Faces", img)
# cv.waitKey(0)

people=['maha','thirithi']
DIR= r'C:\Users\balam\OneDrive\Pictures\thirithi'

features=[]
labels=[]
def create_pic():
    for person in people:
        path = os.path.join(DIR, person)
        label= people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)

            img_arr = cv.imread(img_path)
            gray=cv.cvtColor(img_arr, cv.COLOR_BGR2GRAY)
            faces_rect= haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                faces_roi= gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)
create_pic()
# print(f'length of the features={len(features)}')
# print(f'length of the labels={len(labels)}')
print('training done--------------')


people=['maha','thirithi']
features=np.load('features.npy', allow_pickle=True)
labels=np.load('labels.npy')

features=np.array(features, dtype='object')
labels=np.array(labels)

face_recog=cv.face.LBPHFaceRecognizer_create()
face_recog.train(features,labels)
face_recog.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)

img=cv.imread(r"C:\Users\balam\OneDrive\Pictures\thirthi\Snapchat-41544487.jpg")
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('person', gray)

faces_rect=haar_cascade.detectMultiScale(gray, 1.1, 4)
for (x,y,w,h) in faces_rect:
    faces_roi= gray[y:y+h, x:x+w]
    label, confidence = face_recog.predict(faces_roi)
    print(f'label={people[label]} with a confidence of {confidence}')
    cv.putText(img, str(people[label]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0,(0,255,0), thickness=2)
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)
cv.imshow('Detected face', img)
cv.waitKey(0)
