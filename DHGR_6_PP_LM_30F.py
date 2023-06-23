import cv2
import os
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
from matplotlib import pyplot as plt
import time
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from scipy import stats
import tensorflow as tf

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

# Actions that we try to detect
actions = np.array(['follow', 'stop', 'turn_left', 'turn_right', 'move_forward', 'move_backward'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

# Folder start
start_folder = 0

for action in actions: 
    for sequence in range(1,no_sequences+1):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# Preprocessing data
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])
        

model = Sequential()
model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(30,126)))
model.add(LSTM(256, return_sequences=True, activation='relu'))
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# Load the LSTM model
model.load_weights('DHGR_6_PP_LM_30F.h5')

colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        #cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        #cv2.rectangle(output_frame,(384,0),(510,128),(0,255,0),-1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame

# 1. New detection variables

# Sequence variable will collect the 30 frames from the webcam via OpenCV
sequence = []

# Sentence will concatinate the history of gesture detected.
sentence = []

predictions = []

# Threshold will render the results which are above this threshold
threshold = 0.90

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y, landmark_z]) 
            
    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list)) 

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list)) 
    
    if len(temp_landmark_list)==126: 
        temp_landmark_list
    elif len(temp_landmark_list)==63:
        temp_landmark_list.extend(np.zeros(21*3))
    else:
        temp_landmark_list.extend(np.zeros(21*6))
    
    return temp_landmark_list

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (100, 150, 50), 6)

    return image


# For webcam input:
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
use_brect = True

with mp_hands.Hands(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
      while cap.isOpened():
        success, image = cap.read()
        # Rescale the frame by a factor of 0.5
        #image = cv2.resize(image, None, fx=0.5, fy=0.5)
        
        # Flip the frame horizontally
        image = cv2.flip(image, 1)
        
        debug_image = copy.deepcopy(image)
        if not success:
              print("Ignoring empty camera frame.")
              # If loading a video, use 'break' instead of 'continue'.
              continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        
        handedness = results.multi_handedness
        print(results)
        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
              for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS, 
                                               mp_drawing_styles.get_default_hand_landmarks_style(), 
                                               mp_drawing_styles.get_default_hand_connections_style())
                    # Bounding box calculation
                    brect = calc_bounding_rect(debug_image, hand_landmarks)
                    
                    # Landmark calculation
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    # Conversion to relative coordinates / normalized coordinates
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    
                    '''if len(pre_processed_landmark_list)==126: 
                        continue
                    elif len(pre_processed_landmark_list)==63:
                        pre_processed_landmark_list.extend(np.zeros(21*3))
                    else:
                        pre_processed_landmark_list.extend(np.zeros(21*6))'''
                    
                    # Drawing part
                    image = draw_bounding_rect(use_brect, image, brect)    
        
                    # 2. Prediction logic
                    sequence.append(pre_processed_landmark_list)
                    sequence = sequence[-30:]

                    if len(sequence) == 30:
                        res = model.predict(np.expand_dims(sequence, axis=0))[0]
                        print(actions[np.argmax(res)])
                        predictions.append(np.argmax(res))


                    #3. Viz logic
                        if np.unique(predictions[-20:])[0]==np.argmax(res): 
                            if res[np.argmax(res)] > threshold: 

                                if len(sentence) > 0: 
                                    if actions[np.argmax(res)] != sentence[-1]:
                                        sentence.append(actions[np.argmax(res)])
                                else:
                                    sentence.append(actions[np.argmax(res)])

                        if len(sentence) > 3: 
                            sentence = sentence[-3:]

                        # Viz probabilities
                        image = prob_viz(res, actions, image, colors)
            
        cv2.rectangle(image, (0,0), (2000, 40), (200, 197, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', image) #cv2.flip(image, 1))
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()