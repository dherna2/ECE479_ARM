import numpy as np
import cv2
import threading
import time
import rospy
from std_msgs.msg import Float64

t = .5
x = 0


#this is the cascade we just made. Call what you want
frank_cascade = cv2.CascadeClassifier('./test.xml')
face_cascade = cv2.CascadeClassifier('./facefront.xml')
eye_cascade = cv2.CascadeClassifier('./eye.xml')

arm = 0


cap = cv2.VideoCapture(0)
#cap = cvCreateCameraCapture(1)
#cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, 640)
#cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, 480)
cap.set(3,1920)
cap.set(4,1080)
font = cv2.FONT_HERSHEY_SIMPLEX



def background():

    global arm
    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # add this
        # image, reject levels level weights.
	    #cv2.putText(img,'CPU', (x, y-h), font, 1, (0,255,255), 2)
	    #cv2.putText(img,str((x+w)-x) +"," + str((y+h)-y) ,(x, y), font, .9, (0,255,255), 2)

        frank = frank_cascade.detectMultiScale(gray, 1.2, 3)
        face = face_cascade.detectMultiScale(gray, 1.5, 10)
        eye = eye_cascade.detectMultiScale(gray, 1.5, 10)
        
        # detect frank face
        for (x,y,w,h) in frank:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
            arm = 1
	    cv2.putText(img,"Robot",(x, y), font, .9, (0,255,255), 2)

	    # detect human face
        for (x,y,w,h) in face:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
            arm = 2
	    cv2.putText(img,"Face",(x, y), font, .9, (0,255,255), 2)

	    # detect eyes
        for (x,y,w,h) in eye:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
            arm = 2
	    cv2.putText(img,"Eye",(x, y), font, .9, (0,255,255), 2)

        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.imshow('img',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


def foreground():


    global arm
    rospy.init_node('joint1_controller')

    #rospy stuff
    p1 = rospy.Publisher('/joint1_controller/command', Float64)
    p2 = rospy.Publisher('/joint2_controller/command', Float64)
    p3 = rospy.Publisher('/joint3_controller/command', Float64)
    p4 = rospy.Publisher('/joint4_controller/command', Float64)
    p5 = rospy.Publisher('/joint5_controller/command', Float64)

    while not rospy.is_shutdown():
        if arm == 1:
            p1.publish(Float64(-1))
            p2.publish(Float64(.66))
            p3.publish(Float64(.33))
            p4.publish(Float64(.0))
            rospy.sleep(t)

            p2.publish(Float64(.33))
            p3.publish(Float64(.0))
            p4.publish(Float64(-.33))
            rospy.sleep(t)

            p2.publish(Float64(.0))
            p3.publish(Float64(-.33))
            p4.publish(Float64(-.66))	
            rospy.sleep(t)

            p2.publish(Float64(-.33))
            p3.publish(Float64(-.66))
            p4.publish(Float64(-.33))
            rospy.sleep(t)

            p2.publish(Float64(-.66))
            p3.publish(Float64(-.33))
            p4.publish(Float64(.0))
       
            rospy.sleep(t)

            p2.publish(Float64(-.33))
            p3.publish(Float64(.0))
            p4.publish(Float64(.33))
            rospy.sleep(t)

            p2.publish(Float64(.0))
            p3.publish(Float64(.33))
            p4.publish(Float64(.66))
            rospy.sleep(t)

            p2.publish(Float64(.33))
            p3.publish(Float64(.66))
            p4.publish(Float64(.33))
            rospy.sleep(t)

            arm = 0

        if arm == 2:
            p1.publish(Float64(0))
            p2.publish(Float64(-.2))
            p3.publish(Float64(-2))
            p4.publish(Float64(-1))
            p5.publish(Float64(-1))
            arm = 0





#f = threading.Thread(name='foreground', target=foreground)
#f.daemon = True
#f.start()


b = threading.Thread(name='background', target=background)
b.daemon = True
b.start()

foreground()







