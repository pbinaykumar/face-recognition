import os
import cv2
import face_recognition
import numpy as np
from gtts import gTTS
from playsound import playsound


path= "AllImages"

images=[]

classNames=[]
myList=os.listdir(path)
for cl in myList:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])


def findEncodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        if len(face_recognition.face_encodings(img)) > 0:
            encode=face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
    return encodeList

encodeListKnown=findEncodings(images)

# cap=cv2.VideoCapture(0)
cap=cv2.VideoCapture(0)

nameList=[]
while True:
    success,img=cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgs=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    facesCurFrame=face_recognition.face_locations(imgs)
    encodesCurFrame=face_recognition.face_encodings(imgs,facesCurFrame)

    for  encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches =face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis =face_recognition.face_distance(encodeListKnown,encodeFace)
        matchIndex=np.argmin(faceDis)

        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            name = ''.join(filter(lambda x: not x.isdigit(), name))
            y1,x2,y2,x1=faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

            if name.lower() not in nameList:
                nameList.append(name.lower())
                text_val = 'Hello {}'.format(name.lower())
                obj = gTTS(text=text_val, lang='en', slow=False)
                obj.save("exam.mp3")
                def play():
                    try:
                        playsound("exam.mp3")
                    except Exception as e:
                        print(e)
                        play()
                play()

        else:
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, "name", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('web',img)
    cv2.waitKey(1)

