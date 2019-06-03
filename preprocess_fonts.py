# Written by Alexander Oshin


import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt


def augment_font_data(font_data_dir, plot=False):
    with open(os.path.join(font_data_dir, 'font_data.pickle'), 'rb') as f:
        data = pickle.load(f)

    images = np.array(data['images'])
    labels = np.array(data['labels'])
    new_images = []
    new_labels = []

    for i in range(len(images)):
        img = cv2.bitwise_not(images[i])
        label = labels[i]
        new_images.append(img)
        new_labels.append(label)

        up_shift = np.zeros((28, 28), dtype=np.uint8)
        up_shift[:24] = img[4:]
        new_images.append(up_shift)
        new_labels.append(label)

        down_shift = np.zeros((28, 28), dtype=np.uint8)
        down_shift[4:] = img[:24]
        new_images.append(down_shift)
        new_labels.append(label)

        right_shift = np.zeros((28, 28), dtype=np.uint8)
        right_shift[:, 4:] = img[:, :24]
        new_images.append(right_shift)
        new_labels.append(label)

        left_shift = np.zeros((28, 28), dtype=np.uint8)
        left_shift[:, :24] = img[:, 4:]
        new_images.append(left_shift)
        new_labels.append(label)

        resize = np.zeros((36, 36), dtype=np.uint8)
        resize[4:32, 4:32] = img
        resize = cv2.resize(resize, (28, 28))
        new_images.append(resize)
        new_labels.append(label)

        resize_up = np.zeros((28, 28), dtype=np.uint8)
        resize_up[:24] = resize[4:]
        new_images.append(resize_up)
        new_labels.append(label)

        resize_down = np.zeros((28, 28), dtype=np.uint8)
        resize_down[4:] = resize[:24]
        new_images.append(resize_down)
        new_labels.append(label)

        resize_right = np.zeros((28, 28), dtype=np.uint8)
        resize_right[:, 4:] = resize[:, :24]
        new_images.append(resize_right)
        new_labels.append(label)

        resize_left = np.zeros((28, 28), dtype=np.uint8)
        resize_left[:, :24] = resize[:, 4:]
        new_images.append(resize_left)
        new_labels.append(label)

        if plot:
            fig, axs = plt.subplots(2, 5)
            axs = axs.flatten()
            axs[0].imshow(img, cmap='gray')
            axs[1].imshow(up_shift, cmap='gray')
            axs[2].imshow(down_shift, cmap='gray')
            axs[3].imshow(right_shift, cmap='gray')
            axs[4].imshow(left_shift, cmap='gray')
            axs[5].imshow(resize, cmap='gray')
            axs[6].imshow(resize_up, cmap='gray')
            axs[7].imshow(resize_down, cmap='gray')
            axs[8].imshow(resize_right, cmap='gray')
            axs[9].imshow(resize_left, cmap='gray')
            plt.show()

    new_images = np.array(new_images) / 255.0
    new_labels = np.array(new_labels)
    new_data = {'images': new_images, 'labels': new_labels}
    with open(os.path.join(font_data_dir, 'font_data_augmented.pickle'), 'wb') as f:
        pickle.dump(new_data, f)


if __name__ == '__main__':
    augment_font_data()
