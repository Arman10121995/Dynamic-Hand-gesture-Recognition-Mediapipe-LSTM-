#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraImagePublisherNode:
    def __init__(self):
        rospy.init_node('camera_image_publisher')
        self.bridge = CvBridge()
        self.camera = cv2.VideoCapture(0)
        self.image_pub = rospy.Publisher('camera_image', Image, queue_size=10)

    def capture_and_publish(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            ret, frame = self.camera.read()
            if ret:
                image_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                self.image_pub.publish(image_msg)
            rate.sleep()

if __name__ == '__main__':
    try:
        node = CameraImagePublisherNode()
        node.capture_and_publish()
    except rospy.ROSInterruptException:
        pass
