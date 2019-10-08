# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 13:05:05 2019

@author: arnab
"""

# import the necessary packages
from imutils.video import VideoStream
from imutils.object_detection import non_max_suppression
import argparse
import datetime
import imutils
import time
import cv2
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", default='test.mp4', help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
ap.add_argument("-t", "--thresh", type=int, default=30, help="threshold for detection")
ap.add_argument("-o", "--output", default='output.avi', help="path to output video file")
ap.add_argument("-f", "--fps", type=int, default=20, help="FPS of output video")
ap.add_argument("-c", "--codec", type=str, default="MJPG", help="codec of output video")
ap.add_argument("-mt","--motion_thresh", type=float, default=0.25, 
                help="fraction of the total frame in motion")
ap.add_argument("-s","--supress_output", type=bool, default=False, help="supress the output video")
args = vars(ap.parse_args())

threshold_val = args["thresh"]

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

# otherwise, we are reading from a video file
else:
	vs = cv2.VideoCapture(args["video"])    

motion_thresh = args["motion_thresh"]

#-------------------------------------------------------------------------------------------
# initialize the FourCC, video writer and the dimensions of the frame

fourcc = cv2.VideoWriter_fourcc(*args["codec"])
writer = None
(h, w) = (None, None)

#-------------------------------------------------------------------------------------------
# initialize the first frame in the video stream
firstFrame = None

input_framecount = 0
output_framecount = 0

# loop over the frames of the video
while True:
	# grab the current frame and initialize the occupied/unoccupied
    frame = vs.read()
    frame = frame if args.get("video", None) is None else frame[1]
    input_framecount += 1

	# if the frame could not be grabbed, then we have reached the end of the video
    if frame is None:
        break

	# resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=1000)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

	# if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
        continue
    
	# compute the absolute difference between the current frame and
	# first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, threshold_val, 255, cv2.THRESH_BINARY)[1]

	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    rects = []
	# loop over the contours
    for c in cnts:
		# if the contour is too small, ignore it
        if cv2.contourArea(c) < args["min_area"]:
            continue
        
        rects.append(cv2.boundingRect(c))
    
    # apply non-maximal supression to reduce overlap of multiple frames
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.3)
    
    total_motion_area = 0

    for (xA, yA, xB, yB) in pick:
        # uncomment the following line and comment line 127-128 if you want the detected
        # motion boxes overlain on the written video
#        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
    # determining the total area in motion
        total_motion_area += (xB-xA)*(yB-yA)
        
#-------------------------------------------------------------------------------------------
    #initialize the writer
    if writer is None:
		# store the image dimensions, initialize the video writer,
        (h, w) = frame.shape[:2]
        total_area = h*w
        writer = cv2.VideoWriter(args["output"], fourcc, args["fps"], (w,h), True)
    
    if total_motion_area >= motion_thresh*total_area:
        output = frame
        # write the output frame to file
        writer.write(output)
        output_framecount += 1
        
#-------------------------------------------------------------------------------------------
    if args["supress_output"] is False:
        # comment these 2 line (this for loop) if you want the original frames with motion
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
            
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    
    	# show the frame and record if the user presses a key
        cv2.imshow("Security Feed", frame)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Frame Delta", frameDelta)
        key = cv2.waitKey(1) & 0xFF
    
    	# if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break
    
    firstframe = gray
    
# cleanup the camera and close any open windows
vs.stop() if args.get("video", None) is None else vs.release()
writer.release()
cv2.destroyAllWindows()

print("\n input frame count = ", input_framecount)
print("\n output frame_count = ", output_framecount)