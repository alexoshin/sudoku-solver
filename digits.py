import cv2
import numpy as np
import pickle
import os

# try to get the SVM from running train.py
# try:
#     svm = pickle.load(open("./saved/svm.p", "rb"))
# except (OSError, IOError) as e:
#     import train
#     svm = pickle.load(open("./saved/svm.p", "rb"))

if os.path.isfile("./saved/svm.xml"):
    svm = cv2.ml.SVM_create()
    svm = svm.load("./saved/svm.xml")
else:
    import train
    svm = cv2.ml.SVM_create()
    svm = svm.load("./saved/svm.xml")

if os.path.isfile("./saved/hog.xml"):
    hog = cv2.HOGDescriptor("./saved/hog.xml")
else:
    import train
    hog = cv2.HOGDescriptor("./saved/hog.xml")

print("Loaded HOG descriptor and SVM model")

# read the image
image = cv2.imread("./images/test.png")
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

# find the contours using retrieval mode RETR_TREE or RETR_EXTERNAL
contoured_image, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("Found contours")

# remove the outer contour (just for this specific image)
# del contours[0]

# get bounding rectangle for each contour
rects = [cv2.boundingRect(contour) for contour in contours]

pred = []
for rect in rects:
    # draw all bounding rectangles
    cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = thresh[pt1:pt1+leng, pt2:pt2+leng]

    # Resize the image
    if len(roi) > 0 and len(roi[0]) > 0:
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        # Calculate the HOG features
        roi_hog_fd = hog.compute(roi)
        nbr = svm.predict(np.asarray([roi_hog_fd]))
        cv2.putText(image, str(int(nbr[1])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
        # pred.append(nbr[1][0][0])

print(pred)
cv2.imshow("Final Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()