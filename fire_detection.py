#!/usr/bin/python

import cv2
import numpy as np
import rospy
# from fire_detection.msg import ObjectDetectorData  # Import the custom message type
import math
from std_msgs.msg import Int32


# Publishes angle and distance between camera and fire and the imaginary line to object_detector_data topic
class FireDetectionNode:
    def __init__(self):
        rospy.init_node('fire_detection_node')
        
        # Parameters for object detection
        self.known_distance = 182.88  # distance in cm
        self.known_width = 22.86      # width in cm
        self.focal_length = 707.5     # focal length in pixels
        self.lower = np.array([16, 90, 90], dtype=np.uint8)
        self.upper = np.array([30, 255, 255], dtype=np.uint8)
        self.cap = cv2.VideoCapture(0)
        self.angle_degree = 0.0
        self.pub = rospy.Publisher("imagine_data", Int32, queue_size=10)
        self.pub1 = rospy.Publisher("distance_data", Int32, queue_size=10)
        self.rate = rospy.Rate(10)

    def distance_finder(self, known_width, focal_length, width_in_frame):
        distance =  (known_width * focal_length) / width_in_frame
        return distance
    
    def angle_finder(self, known_width, focal_length, width_in_frame, distance):
        width_cm = width_in_frame  # (known_width * width_in_frame) / focal_length
        imagine_line = math.sqrt(abs(distance**2 - width_cm**2))
        angle_radians = math.acos(imagine_line / distance)
        self.angle_degree = math.degrees(angle_radians)
        return imagine_line
    
    def run(self):
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (700, 500))
            blur = cv2.GaussianBlur(frame,(21,21), 0)
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.lower, self.upper)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            output = cv2.bitwise_and(frame, frame, mask=mask)
            gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
            _ , contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
            for contour in contours:
                (x, y, w, h) = cv2.boundingRect(contour)
                if cv2.contourArea(contour) < 500:
                    continue
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                distance = self.distance_finder(self.known_width, self.focal_length, w)
                # Call angle_finder method
                imagine = self.angle_finder(self.known_width, self.focal_length, w, distance)
                cv2.putText(frame, "Distance: " + str(int(distance)) + "cm", (20, 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0))
                rospy.loginfo("Angle in degrees: " + str(int(self.angle_degree)))
                rospy.loginfo("Distance in cm: " + str(int(distance)))

                self.pub.publish(int(imagine))
                self.pub1.publish(int(distance))

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            self.rate.sleep()
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    node = FireDetectionNode()
    node.run()
