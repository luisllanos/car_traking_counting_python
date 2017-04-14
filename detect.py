import numpy as np
import cv2


backsub = cv2.BackgroundSubtractorMOG()
best_id = 0
i = 0

cap = cv2.VideoCapture(0)

while 1:
	ret, frame = cap.read()

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
				best_id+=1
				rect = cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
				cv2.putText(frame, str(best_id), (x,y-5), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 0, 0), 2)


		print(best_id)

		cv2.imshow("Track", frame)
		cv2.imshow("background sub", fgmask)

	key = cv2.waitKey(10)
	if key == ord('q'):
		break