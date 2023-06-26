#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist

default_face_id = "default_face_id"
current_face_id = ""
current_action = ""

def face_id_callback(msg):
    global current_face_id
    current_face_id = msg.data

def detected_action_callback(msg):
    global current_action
    current_action = msg.data

def perform_action(action):
    twist_pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    twist = Twist()

    if action == "Go to Base":
        rospy.loginfo("Going to base...")
        # Publish instructions to go to the starting location
        # Set linear velocity to move forward 1 meter
        twist.linear.x = 1.0
        twist_pub.publish(twist)

    elif action == "Follow":
        rospy.loginfo("Following the person...")
        # Publish instructions to follow the person while maintaining a 2-meter distance
        # Set linear velocity to move forward
        twist.linear.x = 0.2  # Adjust the velocity as needed
        twist_pub.publish(twist)

    elif action == "Stop":
        rospy.loginfo("Stopping...")
        # Publish instructions to stop the current motion
        # Set linear and angular velocities to 0
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        twist_pub.publish(twist)

    elif action == "Turn Left":
        rospy.loginfo("Turning left...")
        # Publish instructions to turn left at an angle of 15 degrees
        # Set angular velocity to turn left
        twist.angular.z = 0.2618  # 15 degrees in radians
        twist_pub.publish(twist)

    elif action == "Turn Right":
        rospy.loginfo("Turning right...")
        # Publish instructions to turn right at an angle of 15 degrees
        # Set angular velocity to turn right
        twist.angular.z = -0.2618  # -15 degrees in radians
        twist_pub.publish(twist)

    elif action == "Move Forward":
        rospy.loginfo("Moving forward...")
        # Publish instructions to move 1 meter in the forward direction
        # Set linear velocity to move forward
        twist.linear.x = 0.2  # Adjust the velocity as needed
        twist_pub.publish(twist)

    elif action == "Move Backward":
        rospy.loginfo("Moving backward...")
        # Publish instructions to move 1 meter in the backward direction
        # Set linear velocity to move backward
        twist.linear.x = -0.2  # Adjust the velocity as needed
        twist_pub.publish(twist)

def control_node():
    rospy.init_node('control_node', anonymous=True)
    rospy.Subscriber('face_id', String, face_id_callback)
    rospy.Subscriber('detected_action', String, detected_action_callback)
    rate = rospy.Rate(10)  # 10 Hz

    while not rospy.is_shutdown():
        if current_face_id == default_face_id and current_action == "Lock Person":
            rospy.loginfo("Performing actions for locked person...")
            while current_action != "Unlock Person":
                perform_action(current_action)
                rate.sleep()
            rospy.loginfo("Person unlocked!")

        rate.sleep()

if __name__ == '__main__':
    try:
        control_node()
    except rospy.ROSInterruptException:
        pass
