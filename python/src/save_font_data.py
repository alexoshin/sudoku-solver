# Written by Alexander Oshin


import os
import subprocess
import pickle
import cv2
import matplotlib.font_manager as fnt


def save_font_data(data_dir, exclude_dir=None):

    print('Generating font data...')

    exclusions = []
    if exclude_dir is not None:
        try:
            with(open(exclude_dir, 'r')) as f:
                exclusions = f.read().splitlines()
        except Exception as e:
            print(e)

    # Grab all the fonts
    font_names = fnt.findSystemFonts(fontpaths=None, fontext='ttf')

    font_images_dir = os.path.join(data_dir, 'font_images')
    if not os.path.isdir(font_images_dir):
        os.mkdir(font_images_dir)

    labels = []
    images = []
    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        print('Saving font images for digit {:d}...'.format(i))
        save_dir = os.path.join(font_images_dir, str(i))
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        for font in font_names:
            font_name = os.path.basename(font).split('.')[0]
            if font_name not in exclusions:
                path_to_font = font.replace('\\', '/')
                save_name = os.path.normpath(os.path.join(save_dir, str(font_name) + '.png')).replace('\\', '/')
                # Call Imagemagick, which will save a png of the font digit
                popen = subprocess.Popen(['magick', '-pointsize', '28', '-font', path_to_font, 'label:{:d}'.format(i),
                                 '-channel', 'Black', '-gravity', 'center', '-trim', '-bordercolor', 'White',
                                 '-resize', '28x28>', '-resize', '28x28<', '-extent', '28x28', save_name],
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                out, err = popen.communicate()
                if not err:
                    image = cv2.imread(save_name, 0)  # Read image using OpenCV, since this is the format used later
                    images.append(image)
                    labels.append(i)
                elif 'warning' in err:
                    os.remove('./' + save_name)  # Remove generated image if Imagemagick resulted in a warning

    # Save data to a pickle file
    data = {'labels': labels, 'images': images}
    with open(os.path.join(data_dir, 'font_data.pickle'), 'wb') as f:
        pickle.dump(data, f)
    print('Saved font data to file.')


if __name__ == '__main__':
    save_font_data('./data', './exclude.txt')
