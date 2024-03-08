import cv2
import mediapipe as mp
import time
from fer import FER

cap = cv2.VideoCapture("Videos/vid2.mp4")  # start video
ptime = 0
mpfacedetection = mp.solutions.face_detection
mpdraw = mp.solutions.drawing_utils   # draws the  bounding box around the face and the key points
facedetection = mpfacedetection.FaceDetection()
emotion_detector = FER(mtcnn=True)  # Multitask Cascaded Convolutional Neural Network (MTCNN) ml alg for fd

while True:
    success, img = cap.read()
    if not success:
        break

    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # converting image to rgb
    results = facedetection.process(imgrgb)  # storing the output of fd in results
    # print(results)

    if results.detections:  # if we have detected a face
        for detection in results.detections:  # for that detection in result
            # mpdraw.draw_detection(img , detection)
            bboxC = detection.location_data.relative_bounding_box  # creating a bounding box class
            ih, iw, ic = img.shape # initializing the sizes of the img

            bbox= int(bboxC.xmin*iw), int(bboxC.ymin*ih),\
            int(bboxC.width*iw), int(bboxC.height*ih)  # creates bounding box from bbox class and multiply dimensions
            face_img=img[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]  # stores the cropped region of face detected by the face detector
            #emotions=emotion_detector.detect_emotions(face_img)
            emotion , _ =emotion_detector.top_emotion(face_img)  # detects the emotion of the face_img
            mpdraw.draw_detection(img,detection)  # draws the bounding box
            # print(id,detection) # assigning label id for each of the face
            # print(detection.score)  # possibility of the face
            # print(detection.location_data.relative_bounding_box) # location of the face on the vid
            cv2.putText(img, f'Mood:{emotion}', (bbox[0],bbox[1]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (250, 216, 173), 2) # adds the text of the emotion on the top of the box color is bgr format


    ctime = time.time()  # current time
    fps = 1/(ctime - ptime)  # images rendered divided by time
    ptime = ctime  # present time
    cv2.putText(img, f'FPS: { int (fps)}',(0, 20), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Slowing the rate of frames
        break

cap.release()  # exits the capture
cv2.destroyAllWindows()  # closes the output window and free's the resources
