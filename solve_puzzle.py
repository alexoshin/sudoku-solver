# Written by Alexander Oshin


import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from save_font_data import save_font_data
from preprocess_fonts import augment_font_data
from digit_classifier import train_classifier
from image_processing import read_gray_img, extract_puzzle, project_digits
from sudoku_solver import create_exact_cover_matrix, convert_solution
from dancing_links import find_solution


def solve_puzzle(img_dir, save_dir, exclude_file_dir):

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

    puzzle, transformation_matrix = extract_puzzle(img)
    print('Finding solution...')
    exact_cover_matrix = create_exact_cover_matrix(puzzle)
    solution = find_solution(exact_cover_matrix)
    final_solution = convert_solution(solution)

    if final_solution is not None:
        print('Solution found!')
        solved_digits = np.zeros_like(puzzle, dtype=np.uint8)
        for j in range(len(puzzle)):
            for i in range(len(puzzle[0])):
                if puzzle[j, i] == 0:
                    solved_digits[j, i] = final_solution[j][i]
        final_image = project_digits(solved_digits, img, 495, transformation_matrix)
        cv2.imwrite(save_location, final_image)
        print('Saved solved image to file.')
        return final_solution, final_image
    else:
        print('Could not find solution.')
        return None, None


if __name__ == '__main__':
    # img_path = './images/test1.jpg'
    # data_save_path = './data'
    # exclude_file_path = './exclude.txt'
    img_path = sys.argv[1]
    data_save_path = sys.argv[2]
    exclude_file_path = sys.argv[3]
    solved_puzzle, solved_image = solve_puzzle(img_path, data_save_path, exclude_file_path)
    if solved_puzzle is not None:
        # for row in solved_puzzle:
        #     print(row)
        plt.imshow(cv2.cvtColor(solved_image, cv2.COLOR_BGR2RGB))
        plt.show()
