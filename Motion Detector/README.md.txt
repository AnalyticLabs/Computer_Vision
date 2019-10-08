motion_detector.py


Run the file using the following command arguments in one string:
	
	python motion_detector.py --video /path/to/input/in_video.mp4
				  --min-area 400
				  --thresh 25
				  --output /path/to/output/out_video.mp4
				  --fps 20
				  --codec MJPG
				  --motion_thresh 0.25
				  --supress_output False


Primary tunable arguments :- (for the paths and frame reduction)
	--video : input video name and path
	--output : generated video file path
	--motion_thresh : fraction of the total frame in motion. this is the primary threshold used in this detection model, hence tuning this will change the resultant frames. Should be determined after examining the input file. should lie between (0.15-0.3)
	--supress_output : whether or not you want to see the detected boxes while running the code.


Secondary tunable arguments:- (for the motion detection algorithm)
	--min-area : the minimum area of each detected contour to be considered as a motion area
	-- thresh : threshold of change from the initial frame to be detected 
