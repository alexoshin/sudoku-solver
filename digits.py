import sys
import cv2
import numpy as np
import pickle
import os

argv = sys.argv
opts = {}
while argv:
    if argv[0][0] == '-':
        opts[argv[0]] = argv[1]
    argv = argv[1:]
if "-i" in opts:
    filepath = opts["-i"]
else:
    sys.exit("Must specify an image path with -i")

if not os.path.isfile(filepath):
    sys.exit("File does not exist")

# read the image
image = cv2.imread(filepath)
if image is None:
    sys.exit("Image could not be read")

print("Read image")

# try to get the SVM and HOG descriptor

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

print("Loaded SVM model and HOG descriptor")

# convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

black_image = image.copy()
black_image[:] = (0,0,0)
# cv2.imshow("Black Image", black_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# apply gaussian blur
# gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# apply inverted binary threshold
ret, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)

# cv2.imshow("Gray Image", thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# find the contours using retrieval mode RETR_TREE or RETR_EXTERNAL
contoured_image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print("Found contours")

height, width, channels = image.shape

outer_area = height * width
# print(outer_area)
indexes = []
# print(hierarchy)
for h in range(len(hierarchy[0])):
    # print(cv2.contourArea(contours[h]))
    if cv2.contourArea(contours[h]) < (outer_area / 100):
        indexes.append(h)
    # elif hierarchy[0][h][2] == -1 and hierarchy[0][h][3] > 0:
    #     # print(yes)
    #     indexes.append(hierarchy[0][h][3])

# print(indexes)
# print(contours[1])
# remove the outer contour (just for this specific image)
# del contours[0]
wanted = []
for i in indexes:
    wanted.append(contours[i])

# for contour in contours:
#     # print(contour)
#     if contour[2] is -1 and contour[3] is not -1:
#         wanted.append(contours[contour[3]])

digits_only_image = cv2.drawContours(black_image.copy(), wanted, -1, (255,255,255), -1)
cv2.imshow("Only digit contours", digits_only_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

digits_only_image = cv2.cvtColor(digits_only_image, cv2.COLOR_BGR2GRAY)

digit_contoured_image, digit_contours, digit_hierarchy = cv2.findContours(digits_only_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# cv2.imshow("Black Image", black_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# get bounding rectangle for each contour
rects = [cv2.boundingRect(contour) for contour in digit_contours]

pred = []
for rect in rects:
    # print(rect)
    # draw all bounding rectangles
    cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = thresh[pt1:pt1+leng, pt2:pt2+leng]

    # Resize the image
    if len(roi) > 0 and len(roi[0]) > 0:
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        # cv2.imshow("1",roi)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        roi = cv2.erode(roi, (2, 2))
        # roi = cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, (3,3))
        # roi = cv2.dilate(roi, (5, 5))
        # Calculate the HOG features
        roi_hog_fd = hog.compute(roi)
        nbr = svm.predict(np.asarray([roi_hog_fd]))
        cv2.putText(black_image, str(int(nbr[1])), (rect[0], rect[1]+rect[3]),cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 3)
        # pred.append(nbr[1][0][0])

# cv2.imshow("Final Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cv2.imshow("Black Image", black_image)
cv2.waitKey(0)
cv2.destroyAllWindows()