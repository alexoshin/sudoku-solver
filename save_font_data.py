# Written by Alexander Oshin


import os
import glob
import subprocess
import pickle
import cv2


def save_font_data(font_dir, data_dir, exclude_dir):

    # Convert paths to operating system for consistency
    font_dir = os.path.normpath(font_dir)
    data_dir = os.path.normpath(data_dir)

    with(open(exclude_dir, 'r')) as f:
        exclusions = f.read().splitlines()

    # Grab all the fonts
    font_names = []
    for extension in ['*.ttf', '*.ttc', '*.otc']:
        font_names.extend(glob.glob(os.path.join(font_dir, extension)))

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    labels = []
    images = []
    for i in range(10):
        print('Saving font images for digit {:d}'.format(i))
        save_dir = os.path.join(data_dir, str(i))
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        for font in font_names:
            font_name = os.path.basename(font).split('.')[0]
            if font_name not in exclusions:
                path_to_font = font.replace('\\', '/')

                save_name = os.path.normpath(os.path.join(save_dir, str(font_name) + '.png')).replace('\\', '/')

                # Call Imagemagick, which will save a png of the digit
                subprocess.call(['magick', '-pointsize', '28', '-font', path_to_font, 'label:{:d}'.format(i),
                                 '-channel', 'Black', '-gravity', 'center', '-trim', '-bordercolor', 'White',
                                 '-resize', '28x28>', '-resize', '28x28<', '-extent', '28x28', save_name])
                image = cv2.imread(save_name, 0)  # Read image using OpenCV, since this is the format used later
                images.append(image)
                labels.append(i)

    # Save data to a pickle file
    data = {'labels': labels, 'images': images}
    with open(os.path.join(data_dir, 'font_data.pickle'), 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    font_path = 'C:/Windows/Fonts'
    data_path = './font_data'
    exclude_path = './exclude.txt'
    save_font_data(font_path, data_path, exclude_path)
