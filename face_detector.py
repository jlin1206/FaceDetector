import face_recognition
import cv2
import numpy as np
import os

#This makes it so that you do not have to individually encode for each image. But rather that you can just put the images in the KnownFaces folder and it will do all the encodings. 
Library = 'KnownFaces' 
known_faces = []
names = []
myList = os.listdir(Library)
#print(myList)
for i in myList:
    curr = cv2.imread(f'{Library}/{i}')
    names.append(curr)
    known_faces.append(os.path.splitext(i)[0])
print(known_faces)


#function to encode all the images files for later use
def Encodings(images):
    known_face_encodings = [] 
    for i in images:
        i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        face_encoding = face_recognition.face_encodings(i)[0]
        known_face_encodings.append(face_encoding)
    return known_face_encodings

Known_encodings = Encodings(names)
#print(Known_encodings)

capture = cv2.VideoCapture(0)

while True:
    ret, img = capture.read()
    imgsmall = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgsmall = cv2.cvtColor(imgsmall, cv2.COLOR_BGR2RGB)

    Face_curframe = face_recognition.face_locations(imgsmall)
    encodeCurFrame = face_recognition.face_encodings(imgsmall, Face_curframe)

    for encodedFace,faceLocation in zip(encodeCurFrame,Face_curframe):
        matches = face_recognition.compare_faces(Known_encodings,encodedFace)
        facedist = face_recognition.face_distance(Known_encodings,encodedFace)
        matchIndex = np.argmin(facedist)
        print(matchIndex)
        
        if matches[matchIndex]:
            name = known_faces[matchIndex].upper()
        else:
            name = 'Unkown'
        top,right,bottom,left = faceLocation
        top = top * 4
        right = right * 4
        bottom = bottom * 4
        left = left * 4
        cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 4)
        cv2.rectangle(img, (left, bottom - 35), (right, bottom), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_COMPLEX,1,(255, 255, 255), 2)
    
    cv2.imshow('Video', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()
