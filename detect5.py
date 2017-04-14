import numpy as np
import cv2

cap = cv2.VideoCapture(0)
#namedWindow('OptFlow', cv2.CV_WINDOW_AUTOSIZE)

# params for corner detection
feature_params = dict( maxCorners = 100,
				   qualityLevel = 0.3,
				   minDistance = 7,
				   blockSize = 7 )

# LK params
lk_params = dict( winSize  = (15,15),
			  maxLevel = 2,
			  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

while(1):
	ret,frame = cap.read()
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# calculate optical flow
	p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None,    **lk_params)

	# Select good points
	good_new = p1[st==1]
	good_old = p0[st==1]

	#cv2.imshow('OptFlow',img)
	cv2.imshow('OptFlow',frame)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

# update prev frame with new
old_gray = frame_gray.copy()
p0 = good_new.reshape(-1,1,2) 


cv2.destroyAllWindows()
cap.release()
