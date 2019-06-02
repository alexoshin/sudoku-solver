# Written by Alexander Oshin


from digit_classifier import train_classifier
from image_processing import read_gray_img, extract_puzzle
from sudoku_solver import create_exact_cover_matrix, convert_solution
from dancing_links import find_solution


# TODO: Add font data generation
def solve_puzzle(img_dir):
    img = read_gray_img(img_dir)
    puzzle = extract_puzzle(img)
    exact_cover_matrix = create_exact_cover_matrix(puzzle)
    solution = find_solution(exact_cover_matrix)
    solved_puzzle = convert_solution(solution)
    return solved_puzzle


if __name__ == '__main__':
    img_path = './images/test1.jpg'
    puzzle = solve_puzzle(img_path)
    print(puzzle)
