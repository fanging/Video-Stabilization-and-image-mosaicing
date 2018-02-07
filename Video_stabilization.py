import numpy as np 
import cv2
import math

SMOOTHING_RADIUS = 30
HORIZONTAL_BORDER_CROP = 20

cap = cv2.VideoCapture('video.mp4')

ret, prev = cap.read()
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

previous_to_current_transform=[]

k=1

while(True):
    try:
        ret, curr = cap.read( )
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        previous_corner = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
        current_corner, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, previous_corner, None, **lk_params)

        previous_corner2 = previous_corner[st == 1]
        current_corner2 = current_corner[st == 1]

        T = cv2.estimateRigidTransform(previous_corner2, current_corner2, False)
        # if(T.any()):
        #	break
        dx = T[0, 2]
        dy = T[1, 2]
        da = math.atan2(T[1, 0], T[0, 0])

        previous_to_current_transform.append((dx, dy, da))
        prev = curr.copy( )
        prev_gray = curr_gray.copy( )
        cv2.imshow(curr)
        k += 1
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break
    except:
        break;

max_frames=k
a,x,y = 0.0,0.0,0.0
trajectory=[]

for i in range(len(previous_to_current_transform)):
	tx,ty,ta=previous_to_current_transform[i]
	x+=tx
	y+=ty
	a+=ta
	trajectory.append((x,y,a))


smoothed_trajectory=[]

for i in range(len(trajectory)):
    sx,sy,sa,ctr=0.0,0.0,0.0,0
    for j in range(-SMOOTHING_RADIUS,SMOOTHING_RADIUS+1):
    	if(i+j >= 0 and i+j < len(trajectory)):
    		tx,ty,ta=trajectory[i+j]
    		sx+=tx
    		sy+=ty
    		sa+=ta
    		ctr+=1
    smoothed_trajectory.append((sx/ctr,sy/ctr,sa/ctr))

new_previous_to_current_transform=[]
a,x,y=0.0,0.0,0.0

for i in range(len(previous_to_current_transform)):
	tx,ty,ta=previous_to_current_transform[i]
	sx,sy,sa=smoothed_trajectory[i]
	x+=tx
	y+=ty
	a+=ta
	new_previous_to_current_transform.append((tx+sx-x,ty+sy-y,ta+sa-a))


vert_border = HORIZONTAL_BORDER_CROP * len(prev) / len(prev[0])

cap=cv2.VideoCapture('video.mp4')

k=0

while(k < max_frames-1):
        ret, curr = cap.read( )
        tx, ty, ta = new_previous_to_current_transform[k]
        T = np.matrix([[math.cos(ta), -math.sin(ta), tx], [math.sin(ta), math.cos(ta), ty]])
        curr2 = cv2.warpAffine(curr, T, (len(curr), len(curr[0])))
        curr2 = curr2[HORIZONTAL_BORDER_CROP:len(curr2[0] - HORIZONTAL_BORDER_CROP),
                vert_border:len(curr2) - vert_border]
        curr2 = cv2.resize(curr2, (len(curr[0]), len(curr)))

        cv2.imshow(curr2)

        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break
        k += 1

cap.release()
