from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
import pytesseract
import argparse
import time

ap=argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
	help="path to input image")
ap.add_argument("-east", "--east", type=str,
	help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
	help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=320,
	help="resized image height (should be multiple of 32)")
args = vars(ap.parse_args())

image=cv2.imread(args["image"])
original=image.copy()
(H,W)=image.shape[:2]
(newh,neww)=(args["width"],args["height"])
rw=W/float(neww)
rh=H/float(newh)

#now we will resize the image beacause our east detection modul works on 32* pixels imag
image=cv2.resize(image,(neww,newh))
(H,W)=image.shape[:2]

layers=["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"]
#the first layer is a sigmoid activation function that gives us the probability if a text is there or not
#the second layer gives us the geometric dimensions of the text in the image.

#now loading the east text detector
print("[INFO] loading EAST text detector...")
net=cv2.dnn.readNet(args["east"])
blob=cv2.dnn.blobFromImage(image,1.0,(W,H),(123.68, 116.78, 103.94),swapRB=True,crop=False)
start = time.time()
net.setInput(blob)
(scores,geometry)=net.forward(layers) #will return the probablistic score and geometry
end = time.time()
print("[INFO] text detection took {:.6f} seconds".format(end - start))


(nRows,nColumns)=scores.shape[2:4]
#print(nRows)
#print(nColumns)
cord=[] #will store the geometric dimensions of the test
confidence=[] #will store the probabilistic score

for y in range(0,nRows,1):
	scoresData=scores[0,0,y]
	x0=geometry[0,0,y]
	x1=geometry[0,1,y]
	x2=geometry[0,2,y]
	x3=geometry[0,3,y]
	anglesData=geometry[0,4,y]

	for x in range(0,nColumns,1):
		#if our score doesnt matches minimum required confidence we set it will ignore that part
		if scoresData[x]<args["min_confidence"]:
			continue
		#now when we are using the East detector it resizes the image to four time smaller so now we  will give
		#it is original size by multiplying it by four
		(offsetX,offsetY)=(x*4.0,y*4.0)
		angle=anglesData[x] #extracting the rotation angle
		cos=np.cos(angle)
		sin=np.sin(angle)
		#now we will find out the box coordinates
		h= x0[x] + x2[x]
		w= x1[x] + x3[x]
		endX = int(offsetX + (cos * x1[x]) + (sin * x2[x]))
		endY = int(offsetY - (sin * x1[x]) + (cos * x2[x]))
		startX = int(endX - w)
		startY = int(endY - h)

		cord.append((startX, startY, endX, endY))
		confidence.append(scoresData[x])

box=non_max_suppression(np.array(cord),probs=confidence)
for (startX,startY,endX,endY) in box:
	#now we will scale the coordinates of the boxes back to its original size
	startX=int(startX*rw)
	startY=int(startY*rh)
	endX=int(endX*rw)
	endY=int(endY*rh)
	#now we will use the copy image we created at the beginning to draw the bounding box
	cv2.rectangle(original,(startX,startY),(endX,endY),(0,0,255),3)
cv2.imshow("Text Image",original)
cv2.waitKey(0)
