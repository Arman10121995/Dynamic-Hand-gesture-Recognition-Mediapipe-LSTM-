#!/usr/bin/env python

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
import rospy
from std_msgs.msg import String

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def callback(data):
    face_id = data.data
    # Perform action recognition only if face_id matches default face_id
    if face_id == default_face_id:
        detect_action()

# Default Face ID
default_face_id = "default_face_id"

# ROS Node
rospy.init_node('action_recognition_node')

# Publisher for detected actions
action_pub = rospy.Publisher('detected_action', String, queue_size=10)

# Subscriber for Face ID
face_id_sub = rospy.Subscriber('face_id', String, callback)


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                              )
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, lh, rh])


# Actions that we try to detect
actions = np.array(['Lock_person', 'Unlock_person', 'Go_to_base', 'Follow', 'Stop', 'Turn_left', 'Turn_right', 'Move_forward', 'Move_backward'])


model = Sequential()
model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(30, 258)))
model.add(LSTM(256, return_sequences=True, activation='relu'))
model.add(LSTM(128, return_sequences=False, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# Load the LSTM model
model.load_weights('DHGR_9_NPP_LM_30F_hol.h5')

# Set up colors for each action (modify as needed)
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
          (255, 0, 255), (0, 255, 255)]


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    num_actions = len(actions)

    for num, action in enumerate(actions):
        prob = res[num]
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, action, (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

    return output_frame


def detect_action():
    # New detection variables
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.95

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    use_brect = True

    with mp_holistic.Holistic(model_complexity=1, min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            success, image = cap.read()
            debug_image = copy.deepcopy(image)
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image, results = mediapipe_detection(image, holistic)
            print(results)

            draw_styled_landmarks(image, results)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))

                if np.unique(predictions[-20:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 3:
                    sentence = sentence[-3:]

                image = prob_viz(res, actions, image, colors)

            cv2.rectangle(image, (0, 0), (2000, 40), (200, 197, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()


def face_callback(face_id):
    default_face_id = "default_face_id"

    if face_id == default_face_id:
        detect_action()


def main():
    rospy.init_node('action_recognition_node')
    rospy.Subscriber('face_id_topic', String, face_callback)
    rospy.spin()


if __name__ == '__main__':
    main()