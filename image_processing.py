# Written by Alexander Oshin
# References: OpenCV documentation, Wikipedia pages for classical computer vision topics


import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


# Read an image as grayscale
def read_gray_img(filepath):
    gray_img = cv2.imread(filepath, 0)
    if gray_img is None:
        print('Error while reading image')
        exit(1)
    return gray_img


# Calculate the intersection between two lines in Hesse normal form (rho, theta)
def intersection(line1, line2):
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return x0, y0


# Calculate the geometric distance between two points
def dist(point1, point2):
    return (point1[0]-point2[0])**2 + (point1[1]-point2[1])**2


# Combine close points (within some distance d) by finding their average coordinate
def fuse(points, d):
    combined_points = []
    n_points = len(points)
    taken = np.full(n_points, False)
    for i in range(n_points):
        if not taken[i]:
            count = 1
            point = [points[i][0], points[i][1]]
            taken[i] = True
            for j in range(i+1, n_points):
                if dist(points[i], points[j]) < d ** 2:
                    point[0] += points[j][0]
                    point[1] += points[j][1]
                    count += 1
                    taken[j] = True
            point[0] /= count
            point[1] /= count
            combined_points.append((int(np.round(point[0])), int(np.round(point[1]))))
    return combined_points


# Grab the puzzle region and warp the perspective
def extract_puzzle(img):
    height, width = len(img), len(img[0])
    original = img.copy()
    img = cv2.GaussianBlur(img, (9, 9), 0)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 5, 2)
    kernel = np.ones((3, 3), dtype=np.uint8)
    img = cv2.dilate(img, kernel=kernel)
    new_img = np.zeros_like(img, dtype='uint8')
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(new_img, [c], 0, (255, 255, 255), 3)
    lines = cv2.HoughLines(new_img, 1, np.pi/90, 200)
    horizontals = []
    verticals = []
    for line in lines:
        rho, theta = line[0]
        if theta < 1 or theta > 3:
            verticals.append(line)
        else:
            horizontals.append(line)
    points = []
    for h in horizontals:
        for v in verticals:
            points.append(intersection(h, v))
    points = np.float32(fuse(points, 10))
    size = max(height, width)
    arr = np.float32([(size, size), (0, size), (size, 0), (0, 0)])
    M = cv2.getPerspectiveTransform(points, arr)

    undistorted = cv2.warpPerspective(original, M, (size, size))
    dst = cv2.adaptiveThreshold(undistorted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 101, 1)
    # dst = cv2.dilate(undistorted, kernel)
    box_size = int(round(size / 9))
    imgs = np.zeros((9, 9, box_size, box_size), dtype='uint8')
    digits = np.zeros((9, 9), dtype='uint8')
    classifier = load_model('classifier.h5')
    for i in range(9):
        for j in range(9):
            potential_digit = dst[box_size*i+6:box_size*(i+1)-6, box_size*j+6:box_size*(j+1)-6]
            # plt.imshow(potential_digit, cmap='gray')
            # plt.show()
            white_pixel_count = cv2.countNonZero(potential_digit)
            # print(white_pixel_count)
            if white_pixel_count > 150:
                imgs[i, j] = dst[box_size*i:box_size*(i+1), box_size*j:box_size*(j+1)]
                scaled_digit = cv2.resize(potential_digit, (28, 28)) / 255.0
                predicted_value = classifier.predict(np.array(scaled_digit).reshape((1, 28, 28, 1)))
                predicted_value = np.argmax(predicted_value[0])
                predicted_value += 1  # Classes are digits 1-9, but represented as 0-8 in the classifier
                # print(predicted_value)
                # plt.imshow(scaled_digit, cmap='gray')
                # plt.show()
                digits[i, j] = predicted_value
    for row in digits:
        print(row)
    return digits


if __name__ == '__main__':
    img = read_gray_img('./images/test1.jpg')
    puzzle = extract_puzzle(img)
    print(puzzle)
