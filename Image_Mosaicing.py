import cv2
import numpy as np
from numpy.linalg import pinv
from matplotlib import pyplot as plt

def warpInv(p):
	output_inverse=np.matrix([[0.1]]*6)
	div=(1+p[0,0])*(1+p[3,0])-p[1,0]*p[2,0]
	output_inverse[0,0]=(-p[0,0]-p[0,0]*p[3,0]+p[1,0]*p[2,0])/div
	output_inverse[1,0]=(-p[1,0])/div
	output_inverse[2,0]=(-p[2,0])/div
	output_inverse[3,0]=(-p[3,0]-p[0,0]*p[3,0]+p[1,0]*p[2,0])/div
	output_inverse[4,0]=(-p[4,0]-p[3,0]*p[4,0]+p[2,0]*p[5,0])/div
	output_inverse[5,0]=(-p[5,0]-p[0,0]*p[5,0]+p[1,0]*p[4,0])/div
	return output_inverse

def get_New_Coordinate(img1,img2,img1_coord,img2_coord,w,sobelx,sobely):

	y1 = int(img1_coord[0])
	x1 = int(img1_coord[1])
	y2 = int(img2_coord[0])
	x2 = int(img2_coord[1])

	if(((y1+w)>len(img1)) or ((x1+w)>len(img1[0])) or ((y1-w)<0) or ((x1-w)<0) or ((y2+w)>len(img1)) or ((x2+w)>len(img1[0])) or ((y2-w)<0) or ((x2-w)<0)):
		return np.matrix([[1.0,0.0,0.0],[0.0,1.0,0.0]])
	(rows, cols) = img1.shape
	T = img1[y1-w:y1+w, x1-w:x1+w]
	y_array = np.tile(np.arange(rows), (cols,1))[0:2*w,0:2*w].T
	x_array = np.tile(np.arange(cols), (rows,1))[0:2*w,0:2*w]

	xgrad = sobelx[y1-w:y1+w, x1-w:x1+w]
	ygrad = sobely[y1-w:y1+w, x1-w:x1+w]

	steep_descent_img = np.array([np.multiply(xgrad, x_array), np.multiply(ygrad, x_array), np.multiply(xgrad, y_array), np.multiply(ygrad, y_array), xgrad, ygrad])

	hessian = np.tensordot(np.swapaxes(steep_descent_img.T,0,1), steep_descent_img.T, axes=([1,0],[0,1]))

	inv_hessian = pinv(hessian)

	p1,p2,p3,p4,p5,p6=0.0,0.0,0.0,0.0,0.0,0.0
	k=0
	bad_itr=0
	min_cost = float("inf")
	minW=np.matrix([[1.0,0.0,0.0],[0.0,1.0,0.0]])
	W=np.matrix([[1.0,0.0,0.0],[0.0,1.0,0.0]])

	while(k<=10000):
		try:
			k = k + 1
			pos = [[W.dot(np.matrix([[x2 + i - w], [y2 + j - w], [1]], dtype='float')) for i in range(2 * w)] for j in
				   range(2 * w)]
			if not (0 <= (pos[0][0])[0, 0] < cols and 0 <= (pos[0][0])[1, 0] < rows and 0 <= pos[w - 1][0][
				0, 0] < cols and 0 <= pos[w - 1][0][1, 0] < rows and 0 <= pos[0][w - 1][0, 0] < cols and 0 <=
				pos[0][w - 1][1, 0] < rows and 0 <= pos[w - 1][w - 1][0, 0] < cols and 0 <= pos[w - 1][w - 1][
				1, 0] < rows):
				return np.matrix([[-1], [-1]])

			I = np.matrix(
				[[img2[int((pos[i][j])[1, 0]), int((pos[i][j])[0, 0])] for j in range(2 * w)] for i in range(2 * w)])

			error_image = np.absolute(np.matrix(I, dtype='int') - np.matrix(T, dtype='int'))

			steepest_error = np.tensordot(steep_descent_img.T, np.expand_dims(error_image, axis=2),
										   axes=([1, 0], [0, 1]))
			current_cost = np.sum(np.absolute(steepest_error))
			delta_p = np.matmul(inv_hessian, steepest_error)
			dp = warpInv(delta_p)
			p1, p2, p3, p4, p5, p6 = p1 + dp[0, 0] + p1 * dp[0, 0] + p3 * dp[1, 0], p2 + dp[1, 0] + dp[0, 0] * p2 + p4 * \
									 dp[1, 0], p3 + dp[2, 0] + p1 * dp[2, 0] + p3 * dp[3, 0], p4 + dp[3, 0] + p2 * dp[
										 2, 0] + p4 * dp[3, 0], p5 + dp[4, 0] + p1 * dp[4, 0] + p3 * dp[5, 0], p6 + dp[
										 5, 0] + p2 * dp[4, 0] + p4 * dp[5, 0]
			W = np.matrix([[1 + p1, p3, p5], [p2, 1 + p4, p6]])

			if (current_cost <= min_cost):
				min_cost = current_cost
				bad_itr = 0
				minW = W
			else:
				bad_itr += 1

			if (bad_itr == 30):
				W = minW
				return W

			if (np.sum(np.absolute(delta_p)) < 0.0006):
				return W
		except:
			break;
	return W

img1 = cv2.imread("1Hill.jpg",0)
img1 = cv2.resize(img1, dsize = (0,0), fx = 0.5, fy = 0.5)
img2 = cv2.imread("2Hill.jpg",0)
img2 = cv2.resize(img2, dsize = (0,0), fx = 0.5, fy = 0.5)


(rows, cols) = img1.shape
img2 = img2[0:rows,0:cols]
(rows, cols) = img1.shape

orb = cv2.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# draw only keypoints location,not size and orientation
imgorb1 = cv2.drawKeypoints(img1, kp1, None, color=(0,255,0), flags=0)
imgorb2 = cv2.drawKeypoints(img2, kp2, None, color=(0,255,0), flags=0)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 4 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:4],None, flags=2)
plt.imshow(img3),plt.show()

list_kp1 = np.array([kp1[matches[0].queryIdx].pt], dtype = np.float32)
list_kp2 = np.array([kp2[matches[0].trainIdx].pt], dtype = np.float32)

for mat in matches[1:4]:
	a = np.array([kp1[mat.queryIdx].pt], dtype = np.float32)
	b = np.array([kp2[mat.trainIdx].pt], dtype = np.float32)
	list_kp1 = np.concatenate((list_kp1, a))
	list_kp2 = np.concatenate((list_kp2, b))

for i in range(len(list_kp1)):
	pass

	sobelx=cv2.Sobel(img1,cv2.CV_32F,1,0,ksize=5)
	sobely=cv2.Sobel(img1,cv2.CV_32F,0,1,ksize=5)
	W = get_New_Coordinate(img1,img2,list_kp1[i],list_kp2[i],15,sobelx,sobely)

	newPosses=[]
	img = cv2.warpAffine(img2,W,(cols+100,rows+100))

	img_ext = np.zeros([rows+100,cols+100])

	for i in range(rows+100):
		for j in range(cols+100):
			if (i<rows and j<cols):
				if(int(img[i,j])*int(img1[i,j]) == 0):
					img_ext[i,j] = int(img[i,j]) + int(img1[i,j])
				else:
					img_ext[i,j] = (int(img[i,j]) + int(img1[i,j]))/2
			else:
				img_ext[i,j] = int(img[i,j]) 

	plt.figure(4)
	plt.imshow(img_ext, cmap = "gray")
	plt.show()