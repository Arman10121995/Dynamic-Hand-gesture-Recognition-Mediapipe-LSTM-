#!/usr/bin/env python

import rospy
from std_msgs.msg import String
import cv2
import copy
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Set the default face ID to match
default_face_id = "default_face_id"

# Actions that we try to detect
actions = np.array(['Lock_person', 'Unlock_person', 'Go_to_base', 'Follow', 'Stop', 'Turn_left', 'Turn_right', 'Move_forward', 'Move_backward'])

class HandGestureDetectionNode:
    def __init__(self):
        rospy.init_node('hand_gesture_detection_node', anonymous=True)
        
        # ROS topics
        self.face_id_sub = rospy.Subscriber('face_id', String, self.face_id_callback)
        self.detected_action_pub = rospy.Publisher('detected_action', String, queue_size=10)
        
        # Initialize MediaPipe Holistic model
        self.mp_holistic = mp.solutions.holistic.Holistic(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        # Load the pre-trained LSTM model
        self.load_lstm_model()
        
        # Other variables
        self.sequence = []
        self.sentence = []
        self.threshold = 0.95
        
    def load_lstm_model(self):
        self.model = Sequential()
        self.model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(30, 258)))
        self.model.add(LSTM(256, return_sequences=True, activation='relu'))
        self.model.add(LSTM(128, return_sequences=False, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(actions.shape[0], activation='softmax'))
        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        self.model.load_weights('catkin_ws/src/action_recognition/src/DHGR_9_NPP_LM_30F_hol.h5')
    
    def face_id_callback(self, msg):
        face_id = msg.data
        if face_id == default_face_id:
            self.start_gesture_detection()
    
    def start_gesture_detection(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        with self.mp_holistic as holistic:
            while cap.isOpened():
                success, image = cap.read()
                
                debug_image = copy.deepcopy(image)
                
                if not success:
                    rospy.logwarn("Ignoring empty camera frame.")
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                self.draw_styled_landmarks(image, results)
                
                keypoints = self.extract_keypoints(results)
                self.sequence.append(keypoints)
                self.sequence = self.sequence[-30:]
                
                if len(self.sequence) == 30:
                    res = self.model.predict(np.expand_dims(self.sequence, axis=0))[0]
                    action = actions[np.argmax(res)]
                    rospy.loginfo(action)
                    
                    if np.unique(res[-20:])[0] == np.argmax(res):
                        if res[np.argmax(res)] > self.threshold:
                            if len(self.sentence) > 0:
                                if action != self.sentence[-1]:
                                    self.sentence.append(action)
                            else:
                                self.sentence.append(action)
                    
                    if len(self.sentence) > 3:
                        self.sentence = self.sentence[-3:]
                    
                    self.publish_detected_action(' '.join(self.sentence))
                    
                cv2.imshow('MediaPipe Hands', image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def draw_styled_landmarks(self, image, results):
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
        )
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )
    
    def extract_keypoints(self, results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, lh, rh])
    
    def publish_detected_action(self, action):
        msg = String()
        msg.data = action
        self.detected_action_pub.publish(msg)
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = HandGestureDetectionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
