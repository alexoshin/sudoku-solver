# Written by Alexander Oshin
# References: Dancing Links by Donald E. Knuth, Stanford University
# Paper available publicly at: https://www.ocf.berkeley.edu/~jchu/publicportal/sudoku/0011047.pdf


class Data:

    def __init__(self, position=None):
        self.L = self
        self.R = self
        self.U = self
        self.D = self
        self.C = self
        self.position = position


class Column(Data):

    def __init__(self, size=0, name=None):
        super().__init__()
        self.size = size
        self.name = name


# Creates a doubly linked node structure
# based on the exact cover problem matrix
def build_nodes(A):
    num_rows, num_cols = len(A), len(A[0])
    h = Column(name='header')
    for c in range(num_cols):
        new_col = Column(name=c)
        new_col.R = h.R
        new_col.L = h
        h.R = new_col
        h = new_col
    h.R.L = h
    h = h.R
    for r in range(num_rows):
        prev_data = None
        for c in range(num_cols):
            h = h.R
            if A[r][c] == 1:
                new_data = Data((r, c))
                new_data.C = h
                new_data.U = h.U
                h.U.D = new_data
                h.U = new_data
                new_data.D = h
                if prev_data is not None:
                    new_data.R = prev_data.R
                    prev_data.R.L = new_data
                    prev_data.R = new_data
                    new_data.L = prev_data
                prev_data = new_data
                h.size += 1
        h = h.R
    return h


# Covers a column object c (as defined by Knuth)
def cover(c):
    c.R.L = c.L
    c.L.R = c.R
    i = c.D
    while i is not c:
        j = i.R
        while j is not i:
            j.D.U = j.U
            j.U.D = j.D
            j.C.size -= 1
            j = j.R
        i = i.D


# Uncovers a column object c (as defined by Knuth)
def uncover(c):
    i = c.U
    while i is not c:
        j = i.L
        while j is not i:
            j.C.size += 1
            j.D.U = j
            j.U.D = j
            j = j.L
        i = i.U
    c.R.L = c
    c.L.R = c


# Chooses a column that minimizes the branching factor
# by picking the column with the smallest number of 1s
def choose_column(h):
    j = h.R
    s = float('inf')
    while j is not h:
        if j.size < s:
            c = j
            s = j.size
        j = j.R
    return c


# Recursively searches for a solution
def search(O, h, k):
    if h.R == h:
        return O
    c = choose_column(h)
    cover(c)
    r = c.D
    while r is not c:
        O.append(r)
        j = r.R
        while j is not r:
            cover(j.C)
            j = j.R
        if search(O, h, k + 1):
            return O
        r = O.pop()
        c = r.C
        j = r.L
        while j is not r:
            uncover(j.C)
            j = j.L

        r = r.D
    uncover(c)
    return None


# Change solution type from data to row numbers
def convert_solution(O):
    if O is None:
        return None
    solution = [o.position[0] for o in O]
    solution.sort()
    return solution


# Finds a solution for A, which is an exact cover problem matrix
def find_solution(A):
    h = build_nodes(A)
    return convert_solution(search([], h, 0))


if __name__ == '__main__':
    test_matrix = [[0, 0, 1, 0, 1, 1, 0],
                   [1, 0, 0, 1, 0, 0, 1],
                   [0, 1, 1, 0, 0, 1, 0],
                   [1, 0, 0, 1, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 1],
                   [0, 0, 0, 1, 1, 0, 1]]
    solution = find_solution(test_matrix)
    if solution:
        for row in solution:
            print('Row: ', row)
    else:
        print('No solution found.')
