import cv2
import numpy as np
import pickle
import os

# try to get the SVM from running train.py
try:
    hog = pickle.load(open("./saved/hog.p", "rb"))
    svm = pickle.load(open("./saved/svm.p", "rb"))
except (OSError, IOError) as e:
    import train
    hog = pickle.load(open("./saved/hog.p", "rb"))
    svm = pickle.load(open("./saved/svm.p", "rb"))
print("Loaded HOG descriptor and SVM model")

# read the image
image = cv2.imread("./images/digits-classification.jpg")
print("Read image")

# convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# apply gaussian blur
# gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# apply inverted binary threshold
ret, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)

# cv2.imshow("Gray Image", gray_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# find the contours using retrieval mode RETR_TREE
contoured_image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print("Found contours")

# remove the outer contour (just for this specific image)
del contours[0]

# get bounding rectangle for each contour
rects = [cv2.boundingRect(contour) for contour in contours]

for rect in rects:
    # draw all bounding rectangles
    cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)

cv2.imshow("Final Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()