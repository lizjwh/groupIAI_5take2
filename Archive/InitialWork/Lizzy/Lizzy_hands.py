#Sign language interpreter tutorial
# pip install opencv-python
# pip install mediapipe


import cv2 #opencv: backbone of a lot of computer vision models
import mediapipe as mp #pose, face and hand detection
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

# MAIN
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

#webcam input
cap = cv2.VideoCapture(0) #index is the first camera attached to the computer see 'device manager' for bus number
if not cap.isOpened():
    print("Cannot find camera device, check connections")

while cap.isOpened():
    success, image = cap.read() #success is True or False, image is the frame

    #Validation part of this while loop - not sure we need this if statement
    if not success:
        print("Ignoring empty camera frame.")
        continue

    #Now for the main part of this while loop --------------------------------------------------------------
    image.flags.writeable = True
    
    #create image object
    image = mp.Image(image_format=mp.ImageFormat.SRGB,data=image)

    #Create HandLandmarker object
    model_path = 'C:/Users/jk20720/OneDrive - University of Bristol/EDes/Year 3/Intro to AI/group-project-group-5-2024/Lizzy/hand_landmarker.task'
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(base_options=base_options, min_hand_detection_confidence=0.05, min_hand_presence_confidence=0.05, min_tracking_confidence=0.05,num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)    
    detection_result = detector.detect(image)
    
    #display image
    image = draw_landmarks_on_image(image.numpy_view(), detection_result)

    #---------------------------------------------------------------------------------------------------------
    #Flip image to get a selfie-view display
    cv2.imshow('press q to quit', cv2.flip(image, 1))

    #this just loops forever and i can't close the window without stopping the program
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


#Show the last still image
#cv2.imshow("Photo",cv2.flip(image, 1))

