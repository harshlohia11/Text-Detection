from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
import pytesseract
import argparse
import time
from imutils.video import VideoStream
from imutils.video import FPS
import imutils

def decode_geometry(scores,geometry):
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
    return cord,confidence

ap = argparse.ArgumentParser()
ap.add_argument("-east", "--east", type=str, required=True,
	help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
	help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=320,
	help="resized image height (should be multiple of 32)")
args = vars(ap.parse_args())

(W,H)=(None,None)
(neww,newh)=(args["width"],args["height"])
(rw,hw)=(None,None)

layers=["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"]
#the first layer is a sigmoid activation function that gives us the probability if a text is there or not
#the second layer gives us the geometric dimensions of the text in the image.

#now loading the east text detector
print("[INFO] loading EAST text detector...")
net=cv2.dnn.readNet(args["east"])
vs=VideoStream(src=0).start()
time.sleep(1.0)
fps=FPS().start()
while True:
    frame=vs.read()
    if frame is None:
        break
    frame = imutils.resize(frame, width=1000)
    original=frame.copy()
    if W is None or H is None:
        H,W=frame.shape[:2]
        rw=W/float(neww)
        rh=H/float(newh)

    frame=cv2.resize(frame,(neww,newh))
    blob = cv2.dnn.blobFromImage(frame, 1.0, (neww, newh),(123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry)=net.forward(layers)
    (cord,confidence)=decode_geometry(scores,geometry)
    box=non_max_suppression(np.array(cord),probs=confidence)
    for (startX,startY,endX,endY) in box:
        #now we will scale the coordinates of the boxes back to its original size
        startX=int(startX*rw)
        startY=int(startY*rh)
        endX=int(endX*rw)
        endY=int(endY*rh)
        #now we will use the copy image we created at the beginning to draw the bounding box
        cv2.rectangle(original,(startX,startY),(endX,endY),(0,0,255),3)
    fps.update()
    cv2.imshow("Text detection",original)
    key=cv2.waitKey(1) & 0xFF
    if key==ord('q'):
        break
fps.stop()
vs.stop()
cv2.destroyAllWindows()