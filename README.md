# Sudoku Solver

## Introduction
A sudoku puzzle solver implemented in Python, utilizing foundational knowledge of algorithms, computer vision, and machine learning.

Given an input image, the processing pipeline is defined as follows:
1. Find the puzzle
2. Recognize the digits
3. Solve the puzzle

## 1. Find the puzzle
The puzzle is found by processing the input image using classical computer vision techniques.  Implementation utilizes the open-source library OpenCV<sup>[1](#opencv)</sup>.

## 2. Recognize the digits
Supervised learning is used to train a model that classifies font digits. This classifier is used to recognize the value of each of the individual digits of the puzzle.

## 3. Solve the puzzle
An exact cover problem representation of the sudoku puzzle is created using the digits and corresponding constraints. Then, the puzzle is solved using a "dancing links" approach, a technique proposed by Donald E. Knuth<sup>[2](#knuth)</sup> to efficiently solve exact cover problems.

## References
<a name="opencv">1</a>: https://opencv.org/  
<a name="knuth">2</a>: https://www.ocf.berkeley.edu/~jchu/publicportal/sudoku/0011047.pdf