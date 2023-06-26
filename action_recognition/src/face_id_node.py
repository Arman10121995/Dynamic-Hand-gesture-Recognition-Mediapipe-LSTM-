#!/usr/bin/env python
import rospy
from std_msgs.msg import String

def face_id_publisher():
    rospy.init_node('face_id_publisher', anonymous=True)
    face_id_pub = rospy.Publisher('face_id', String, queue_size=10)
    rate = rospy.Rate(1)  # 1 Hz

    while not rospy.is_shutdown():
        face_id_pub.publish('default_face_id')
        rate.sleep()

if __name__ == '__main__':
    try:
        face_id_publisher()
    except rospy.ROSInterruptException:
        pass
