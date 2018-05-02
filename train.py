import cv2
import numpy as np
import pickle
import os
from sklearn import datasets

# size based on MNIST digit size
IMAGE_SIZE = 28

# reshape an image to IMAGE_SIZE
def reshape(img):
    return img.reshape(IMAGE_SIZE, IMAGE_SIZE)

# deskew an image; should be grayscale
def deskew(img):
    # get the moments of img
    m = cv2.moments(img)
    # not sure why second order in y is used
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    # finds amount of vertical skew
    skew = m['mu11']/m['mu02']
    # affine transform calculation
    M = np.float32([[1, skew, -0.5*IMAGE_SIZE*skew], [0, 1, 0]])
    # apply affine transformation
    img = cv2.warpAffine(img, M, (IMAGE_SIZE, IMAGE_SIZE), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

# use sklearn ML MNIST dataset; saved in pickle file after first iteration
try:
    mnist = pickle.load(open("./saved/mnist.p", "rb"))
except (OSError, IOError) as e:
    mnist = datasets.fetch_mldata("MNIST Original")
    pickle.dump(mnist, open("./saved/mnist.p", "wb"))
print("Loaded MNIST dataset")

# digits = mnist.data
# digits = np.array(mnist.data, 'int16')
digits = np.asarray(list(map(reshape, mnist.data)))
# labels = np.array(mnist.target, 'int')
labels = np.asarray(mnist.target, 'int')

# randomize the data
rand = np.random.RandomState(10)
shuffle = rand.permutation(len(digits))
digits, labels = digits[shuffle], labels[shuffle]
print("Randomized data")

# test if we reshaped images properly
# cv2.imshow('test', digits[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# deskew all the digits
deskewed_digits = np.asarray(list(map(deskew, digits)))
print("Deskewed all digits")

# test if deskewing worked
# cv2.imshow('test2', deskewed_digits[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# HOG parameters
win_size = (28, 28)
cell_size = (14, 14)
block_size = (28, 28) # may be unnecessary
block_stride = (14, 14)
nbins = 9
deriv_aperture = 1
win_sigma = -1
histogram_norm_type = 0
L2_hys_threshold = 0.2
gamma_correction = 1
nlevels = 64
use_signed_gradients = True

# make a new HOG descriptor with the above parameters
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins,
        deriv_aperture, win_sigma, histogram_norm_type, L2_hys_threshold,
        gamma_correction, nlevels, use_signed_gradients)
# pickle.dump(hog, open("./saved/hog.p", "wb"))
hog.save("./saved/hog.xml")
print("Created HOG descriptor")

# calculate HOG feature descriptor for each image
hog_descriptors = np.asarray(list(map(lambda x: hog.compute(x), deskewed_digits)))
# hog_descriptors = []
# for img in deskewed_digits:
#     hog_descriptors.append(hog.compute(img))
# hog_descriptors = np.squeeze(hog_descriptors)
print("Computed all descriptors")


# split into train and test
split_index = int(0.9 * len(hog_descriptors))
digits_train, digits_test = np.split(deskewed_digits, [split_index])
descriptors_train, descriptors_test = np.split(hog_descriptors, [split_index])
labels_train, labels_test = np.split(labels, [split_index])
print("Split data into training and testing")

# compute the accuracy of a model's prediction
def accuracy(test, prediction):
    count = 0
    for i in range(len(test)):
        if (test[i] == prediction[i]):
            count = count + 1
    return count/len(test)

# make a new SVM model
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_RBF) # Radial Basis Function

# print("Testing configurations...")
# for C in [0.1, 1, 10, 100, 1000]:
#     for gamma in [0.1, 0.5, 0.7, 1, 1.5, 2, 5, 10]:
#         svm.setC(C)
#         svm.setGamma(gamma)
#         svm.train(descriptors_train, cv2.ml.ROW_SAMPLE, labels_train)
#         prediction = svm.predict(descriptors_test)
#         print(C, gamma, 100*accuracy(labels_test, prediction[1]))

# print("Tested SVM model configurations")

# found these after very simple testing; could be tweaked further
svm.setC(9)
svm.setGamma(10)
print("Training model...")
# svm.trainAuto(descriptors_train, cv2.ml.ROW_SAMPLE, labels_train)
svm.train(descriptors_train, cv2.ml.ROW_SAMPLE, labels_train)
print("Trained SVM model")

# svm.save("./saved/svm.xml")
# svm2 = cv2.ml.SVM_create()
# svm2 = svm2.load("./saved/svm.xml")

# predict on test data
prediction = svm.predict(descriptors_test)
print("Current model has %.2f %% accuracy on test data" % (100*accuracy(labels_test, prediction[1])))

# save model
# pickle.dump(svm, open("./saved/svm.p", "wb"))
svm.save("./saved/svm.xml")
print("Saved model")