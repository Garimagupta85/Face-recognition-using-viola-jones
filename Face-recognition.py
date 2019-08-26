 
import cv2

Face_cascade = cv2.CascadeClassifier("/home/garima/Computer-vision/haarcascade_frontalface.xml")
Eye_cascade = cv2.CascadeClassifier("/home/garima/Computer-vision/haarcascade_eye.xml")

def Detect(grayframe, frame):
	Face = Face_cascade.detectMultiScale(grayframe,1.3,5)
	for x,y,w,h in Face:
		cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,0), 3)
		eyeframe = frame[x:x+w,y:y+h]
		eyegrayframe = grayframe[x:x+w,y:y+h]
		Eye = Eye_cascade.detectMultiScale(eyegrayframe, 1.3, 6)
		for ex,ey,ew,eh in Eye:
			cv2.rectangle(eyeframe, (ex,ey), (ex+ew,ey+eh), (0,255,255), 2)
	return frame

video = cv2.VideoCapture(0)

while True:
	tmp, image = video.read()
	image = cv2.flip(image, 1)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	detect = Detect(gray, image)
	cv2.imshow("VIDEO",detect)
	k = cv2.waitKey(0)
	if k == 27:
		break

video.release()
cv2.destroyAllWindows()
