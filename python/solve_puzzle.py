# Written by Alexander Oshin


import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.save_font_data import save_font_data
from src.preprocess_fonts import augment_font_data
from src.digit_classifier import train_classifier
from src.image_processing import read_gray_img, extract_puzzle, project_digits
from src.sudoku_solver import create_exact_cover_matrix, convert_solution
from src.dancing_links import find_solution


def solve_puzzle(img_dir, save_dir, exclude_file_dir, plot_solution=True):

    img = read_gray_img(img_dir)
    dir_name = os.path.dirname(img_dir)
    basename = os.path.basename(img_dir).split('.')[0]
    save_location = os.path.join(dir_name, basename) + '_solved.png'
    save_location = save_location.replace('\\', '/')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    try:
        open(os.path.join(save_dir, 'font_data.pickle'), 'rb')
    except Exception as e:
        print(e)
        save_font_data(save_dir, exclude_file_dir)

    try:
        open(os.path.join(save_dir, 'font_data_augmented.pickle'), 'rb')
    except Exception as e:
        print(e)
        augment_font_data(save_dir)

    try:
        open(os.path.join(save_dir, 'classifier.h5'), 'r')
    except Exception as e:
        print(e)
        train_classifier(os.path.join(save_dir, 'font_data_augmented.pickle'), save_dir)

    puzzle, transformation_matrix = extract_puzzle(img, save_dir)
    print('Finding solution...')
    exact_cover_matrix = create_exact_cover_matrix(puzzle)
    solution = find_solution(exact_cover_matrix)
    final_solution = convert_solution(solution)

    if final_solution is not None:
        print('Solution found!')
        print(np.array(final_solution))
        solved_digits = np.zeros_like(puzzle, dtype=np.uint8)
        for j in range(len(puzzle)):
            for i in range(len(puzzle[0])):
                if puzzle[j, i] == 0:
                    solved_digits[j, i] = final_solution[j][i]
        final_image = project_digits(solved_digits, img, 495, transformation_matrix)
        cv2.imwrite(save_location, final_image)
        print('Saved solved image to file.')
        if plot_solution:
            print('Plotting solution.')
            plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
            plt.xticks([])
            plt.yticks([])
            plt.show()
        return final_solution, final_image
    else:
        print('Could not find solution.')
        return None, None


if __name__ == '__main__':
    img_path = sys.argv[1]
    if len(sys.argv) >= 3:
        data_save_path = sys.argv[2]
    else:
        data_save_path = './data'
    if len(sys.argv) >= 4:
        exclude_file_path = sys.argv[3]
    else:
        exclude_file_path = None
    solved_puzzle, solved_image = solve_puzzle(img_path, data_save_path, exclude_file_path, plot_solution=True)
    print('Done!')
