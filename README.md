# Sudoku Solver

## Introduction
A sudoku puzzle solver implemented in Python, utilizing foundational knowledge of algorithms, complexity theory, computer vision, and machine learning.

My development process is documented in the blog post [here](https://alexoshin.github.io/posts/sudoku-solver/).

## Requirements
* Python 3
    * numpy
    * opencv-python
    * matplotlib
    * tensorflow
* ImageMagick

Run ```pip install -r requirements.txt``` in a new virtual environment to install all required packages with versions that are verified to work.

## Usage

```python python/solve_puzzle.py <image_file> <save_directory> <font_exclude_file>```

```<image_file>``` points to an image of a Sudoku puzzle. This has been verified to support .jpg and .png files, but other image formats may be used at your own risk.

```<save_directory>``` and ```<font_exclude_file>``` are optional.

```<save_directory>``` points to a folder where the font data and classifier will be saved. This defaults to ```./data```.

```<font_exclude_file>``` points to a text file of names of fonts that should be excluded, such as wingding, webdings, etc. An example has been provided in ```exclude.txt```. Since the fonts that are grabbed are dependent on your system, please check the generated font data, and if any digits look weird, add the name to this exclude file. Otherwise, no fonts will be excluded, which can influence the classifier's results.

## The Algorithm

Given an input image, the processing pipeline is defined as follows:
1. Find the puzzle
2. Recognize the digits
3. Solve the puzzle

### 1. Find the puzzle
The puzzle is found by processing the input image using classical computer vision techniques.  Implementation utilizes the open-source library OpenCV<sup>[1](#opencv)</sup>.

This step contains the following substeps:

1. Perform filtering on the image to reduce noise
2. Threshold the image
3. Extract the largest contour in the image
4. Find the best possible square from this contour
5. Perform a perspective transform from the square in the image to a perfect square
6. Divide this new image into 9x9 squares
7. Center each square around the largest white region
8. Separate squares that contain digits from squares that are empty

### 2. Recognize the digits
Supervised learning is used to train a model that classifies font digits. This classifier is used to recognize the value of each of the individual digits of the puzzle. The classifier was created and trained using the open-source library Tensorflow<sup>[2](#tensorflow)</sup>.

This step contains the following substeps:

1. Compile a dataset of font digits
2. Augment this dataset to account for real-world noise and variation
3. Train a classifier on this augmented dataset
4. Use the classifer to predict the digit of each of the squares that contain a digit

### 3. Solve the puzzle
An exact cover problem representation of the sudoku puzzle is created using the digits and corresponding constraints. Then, the puzzle is solved using a "dancing links" approach, a technique proposed by Donald E. Knuth<sup>[3](#knuth)</sup> to efficiently solve exact cover problems.

This step contains the following substeps:

1. Using the digits obtained from the previous step, map the puzzle configuration to an exact cover problem representation
2. Solve the exact cover problem using a "dancing links" approach
3. Map the solution back into Sudoku puzzle space
3. Using the inverse perspective transform, draw the solved digits onto the original image

## References
<a name="opencv">1</a>: https://opencv.org/  
<a name="tensorflow">2</a>: https://www.tensorflow.org/  
<a name="knuth">3</a>: https://www.ocf.berkeley.edu/~jchu/publicportal/sudoku/0011047.pdf  
