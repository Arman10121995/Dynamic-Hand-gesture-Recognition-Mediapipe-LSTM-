#!/usr/bin/env python
import rospy
from std_msgs.msg import String
import subprocess

default_face_id = "default_face_id"
current_face_id = ""

def face_id_callback(msg):
    global current_face_id
    current_face_id = msg.data

def action_detection():
    # Run the action_detection.py file
    process = subprocess.Popen(["python", "DHGR_9_NPP_LM_30F_hol.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    if process.returncode != 0:
        rospy.logerr(f"Error running action_detection.py: {error}")
        return ""

    # Extract the detected action from the output
    detected_action = output.strip().decode()

    return detected_action

def hand_gesture_recognition_node():
    rospy.init_node('hand_gesture_recognition_node', anonymous=True)
    rospy.Subscriber('face_id', String, face_id_callback)
    detected_action_pub = rospy.Publisher('detected_action', String, queue_size=10)
    rate = rospy.Rate(10)  # 10 Hz

    while not rospy.is_shutdown():
        if current_face_id == default_face_id:
            detected_action = action_detection()
            rospy.loginfo(f"Detected Action: {detected_action}")
            detected_action_pub.publish(detected_action)

        rate.sleep()

if __name__ == '__main__':
    try:
        hand_gesture_recognition_node()
    except rospy.ROSInterruptException:
        pass
