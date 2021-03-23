import face_recognition
import cv2
import numpy as np
import os


Library = 'KnownFaces'
known_faces = []
names = []
myList = os.listdir(Library)
print(myList)
for i in myList:
    curr = cv2.imread(f'{Library}/{i}')
    known_faces.append(curr)
    names.append(os.path.splitext(i)[0])
print(names)


def Encodings(known_faces):
    known_face_encodings = [] 
    for i in known_faces:
        i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        face_encoding = face_recognition.face_encodings(i)[0]
        known_face_encodings.append(face_encoding)
    return known_face_encodings

Known_encodings = Encodings(known_faces)
print(Known_encodings)

while True:
    ret, frame = video_capture.read()
    small_
'''
import face_recognition
import cv2
import numpy as np

video_capture = cv2.VideoCapture(0)

Elon_image = face_recognition.load_image_file("Images/elon-musk.jpg")
Elon_face_encoding = face_recognition.face_encodings(Elon_image)[0]

Bill_image = face_recognition.load_image_file("Images/Bill.jpg")
Bill_face_encoding = face_recognition.face_encodings(Bill_image)[0]


known_face_encodings = [
    Elon_face_encoding,
    Bill_face_encoding
]
known_face_names = [
    "Elon",
    "Bill"
]


face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
  
    ret, frame = video_capture.read()


    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

 
    rgb_small_frame = small_frame[:, :, ::-1]


    if process_this_frame:
       
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
      
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

          
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame



    for (top, right, bottom, left), name in zip(face_locations, face_names):
      
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

     
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)


        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
'''