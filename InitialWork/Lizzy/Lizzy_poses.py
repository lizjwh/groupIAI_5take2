#Sign language interpreter tutorial
# pip install opencv-python 
# pip install mediapipe

import cv2 #opencv: backbone of a lot of computer vision models
import mediapipe as mp #pose, face and hand detection

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

#webcam input
cap = cv2.VideoCapture(0) #index is the first camera attached to the computer see 'device manager' for bus number
if not cap.isOpened():
    print("Cannot find camera device, check connections")

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic: #create a holistic object
    while cap.isOpened():
        success, image = cap.read() #success is True or False, image is the frame

        #Validation part of this while loop - not sure we need this if statement
        if not success:
            print("Ignoring empty camera frame.")
            continue

        #Now for the main part of this while loop
        #To optimise, mark the image as not writeable to pass by reference
        image.flags.writeable = False       
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)   #image is processed using this holistic model to give a result

        #This is the prettyprint: draw landmark annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, 
                                  results.face_landmarks, 
                                  mp_holistic.FACEMESH_CONTOURS, 
                                  landmark_drawing_spec=None, 
                                  connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(image,
                                  results.pose_landmarks,
                                  mp_holistic.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        print(results.pose_landmarks)
        #Flip image to get a selfie-view display
        cv2.imshow('press q to quit', cv2.flip(image, 1))
        #this just loops forever and i can't close the window without stopping the program
        if cv2.waitKey(1) == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()


#Show the last still image
#cv2.imshow("Photo",cv2.flip(image, 1))

