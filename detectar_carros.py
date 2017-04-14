import numpy as np
import cv2
from pyimagesearch import imutils


backsub = cv2.BackgroundSubtractorMOG()
best_id = 0
i = 0

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('00002.mov')
#cap = cv2.VideoCapture('FroggerHighway.mp4.part')



while 1:
	ret, frame = cap.read()

	#####
	#cv2.line(frame,(0,170),(300,170),(255,0,255),4)
	#cv2.line(frame,(0,150),(2000,150),(255,0,255),4)
	cv2.line(frame,(0,500),(2000,500),(255,0,255),4)
	frame = imutils.resize(frame, width = 450)
	if ret:
		fgmask = backsub.apply(frame, None, 0.01)
		contours, hierarchy = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

		try: 
			hierarchy = hierarchy[0]
		except: 
			hierarchy = []

		for contour, hierarchy in zip(contours,hierarchy):
			(x,y,w,h) = cv2.boundingRect(contour)
			
			if w > 25 and h > 25:
				#best_id+=1
				rect = cv2.rectangle(frame, (x,y), (x+w,y+h), (180, 1, 0), 1)

				####
				x1=w/2      
				y1=h/2
				cx=x+x1
				cy=y+y1
				centroid=(cx,cy)
				cv2.circle(frame,(int(cx),int(cy)),4,(0,255,0),-1)

				if cy==150:   
					#counter=counter+1
					best_id+=1
					
				cv2.putText(frame, str(best_id), (x,y-5), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 0, 0), 2)


		print(best_id)

		cv2.imshow("Track", frame)
		cv2.imshow("background sub", fgmask)

	key = cv2.waitKey(10)
	if key == ord('q'):
		break
