# Written by Alexander Oshin
# References: Wikipedia pages for Sudoku and Exact Cover, as well as Knuth's paper available here:
# https://www.ocf.berkeley.edu/~jchu/publicportal/sudoku/0011047.pdf


import src.dancing_links


# Creates an exact cover problem matrix from an NxN sudoku puzzle matrix
def create_exact_cover_matrix(puzzle):
    N = len(puzzle)
    sqrt_N = int(N**(1.0 / 2))
    num_rows, num_cols = N**3, N**2 * 4
    exact_cover = [[0 for c in range(num_cols)] for r in range(num_rows)]
    for r in range(N):
        for c in range(N):
            if puzzle[r][c] != 0:
                cands = [puzzle[r][c] - 1]
            else:
                cands = range(N)
            for cand in cands:
                row = r * N**2 + c * N + cand
                row_col = N * r + c
                row_num = N * N + N * r + cand
                col_num = 2 * N * N + N * c + cand
                box_num = 3 * N * N + r // sqrt_N * N * sqrt_N + c // sqrt_N * N + cand
                exact_cover[row][row_col] = 1
                exact_cover[row][row_num] = 1
                exact_cover[row][col_num] = 1
                exact_cover[row][box_num] = 1
    return exact_cover


# Converts a solution in exact cover space back to sudoku space
def convert_solution(solution):
    if solution is None:
        return None
    N = int(len(solution) ** (1.0 / 2))
    sudoku_solution = [[0 for i in range(N)] for j in range(N)]
    for i in range(len(solution)):
        cand = solution[i] % N + 1
        sudoku_solution[i // N][i % N] = cand
    return sudoku_solution


if __name__ == '__main__':
    # test_puzzle = [[4, 0, 0, 0],
    #                [0, 0, 3, 0],
    #                [0, 1, 0, 0],
    #                [0, 0, 0, 2]]
    test_puzzle = [[0, 5, 3, 2, 0, 7, 0, 0, 8],
                   [6, 0, 1, 5, 0, 0, 0, 0, 2],
                   [2, 0, 0, 9, 1, 3, 0, 5, 0],
                   [7, 1, 4, 6, 9, 2, 0, 0, 0],
                   [0, 2, 0, 0, 0, 0, 0, 6, 0],
                   [0, 0, 0, 4, 5, 1, 2, 9, 7],
                   [0, 6, 0, 3, 2, 5, 0, 0, 9],
                   [1, 0, 0, 0, 0, 6, 3, 0, 4],
                   [8, 0, 0, 1, 0, 9, 6, 7, 0]]
    exact_cover_matrix = create_exact_cover_matrix(test_puzzle)
    solution = dancing_links.find_solution(exact_cover_matrix)
    sudoku_solution = convert_solution(solution)
    for row in sudoku_solution:
        print(row)
