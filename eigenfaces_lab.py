# version code c2eb1c41017f+
# Please fill out this stencil and submit using the provided submission script.

from vec import Vec
from vec import dot
from mat import Mat

import svd
import matutil

import eigenfaces

## Task 1

# see documentation of eigenfaces.load_images
raw_images = eigenfaces.load_images('./faces')
face_images = dict((i, Vec({(x, y) for x in range(166) for y in range(189)}, 
    dict(((x, y), raw_images[i][y][x]) for x in range(166) for y in range(189)))) for i in raw_images)

## Task 2

centroid = sum(face_images.values()) / len(face_images)
centered_face_images = dict((k, face_images[k] - centroid) for k in face_images)

## Task 3

A = matutil.rowdict2mat(centered_face_images) # centered image vectors

U, sig, V = svd.factor(A)
orthonormal_basis = matutil.rowdict2mat(list(matutil.mat2coldict(V).values())[:10])

## Task 4

#This is the "transpose" of what was specified in the text.
#Follow the spec given here.
def projected_representation(M, x):
    '''
    Input:
        - M: a matrix with orthonormal rows with M.D[1] == x.D
        - x: a vector
    Output:
        - the projection of x onto the row-space of M
    Examples:
        >>> from vecutil import list2vec
        >>> from matutil import listlist2mat
        >>> x = list2vec([1, 2, 3])
        >>> M = listlist2mat([[1, 0, 0], [0, 1, 0]])
        >>> projected_representation(M, x)
        Vec({0, 1},{0: 1, 1: 2})
        >>> M = listlist2mat([[3/5, 1/5, 1/5], [0, 2/3, 1/3]])
        >>> projected_representation(M, x)
        Vec({0, 1},{0: 1.6, 1: 2.333333333333333})
    '''
    rowdicts = matutil.mat2rowdict(M)
    return Vec(M.D[0], dict((k, dot(rowdicts[k], x)) for k in M.D[0]))

# print(projected_representation(eigenfaces.test_M, eigenfaces.test_x))
## Task 5

#This is the "transpose" of what was specified in the text.
#Follow the spec given here.
def projection_length_squared(M, x):
    '''
    Input:
        - M: matrix with orthonormal rows with M.D[1] == x.D
        - x: vector
    Output:
        - the square of the norm of the projection of x into the
          row-space of M
    Example:
        >>> from vecutil import list2vec
        >>> from matutil import listlist2mat
        >>> x = list2vec([1, 2, 3])
        >>> M = listlist2mat([[1, 0, 0], [0, 1, 0]])
        >>> projection_length_squared(M, x)
        5
        >>> M = listlist2mat([[3/5, 1/5, 1/5], [0, 2/3, 1/3]])
        >>> projection_length_squared(M, x)
        5.644424691358024
    '''
    coord = projected_representation(M, x)
    trans = coord * M
    return dot(trans, trans)

## Task 6

#This is the "transpose" of what was specified in the text.
#Follow the spec given here.
def distance_squared(M, x):
    '''
    Input:
        - M: matrix with orthonormal rows with M.D[1] == x.D
        - x: vector
    Output:
        - the square of the distance from x to the row-space of M
    Example:
        >>> from vecutil import list2vec
        >>> from matutil import listlist2mat
        >>> x = list2vec([1, 2, 3])
        >>> M = listlist2mat([[1, 0, 0], [0, 1, 0]])
        >>> distance_squared(M, x)
        9
        >>> M = listlist2mat([[3/5, 1/5, 1/5], [0, 2/3, 1/3]])
        >>> distance_squared(M, x)
        8.355575308641976
    '''
    return dot(x, x) - projection_length_squared(M, x)

## Task 7

raw_images_test = eigenfaces.load_images('./unclassified', 11)
test_images = dict((i, Vec({(x, y) for x in range(166) for y in range(189)}, 
    dict(((x, y), raw_images_test[i][y][x]) for x in range(166) for y in range(189)))) for i in raw_images_test)

centered_test_images = dict((k, test_images[k] - centroid) for k in test_images)
distances_to_subspace = [distance_squared(orthonormal_basis, centered_test_images[i]) for i in range(len(centered_test_images))]

## Task 8

classified_as_faces = {1, 2, 3, 4, 5} # of dictionary keys

## Task 9

threshold_value = 4e7

## Task 10

#This is the "transpose" of what was specified in the text.
#Follow the spec given here.
def project(M, x):
    '''
    Input:
        - M: an orthogonal matrix with row-space equal to x's domain
        - x: a Vec
    Output:
        - the projection of x into the column-space of M
    Example:
        >>> from vecutil import list2vec
        >>> from matutil import listlist2mat
        >>> x = list2vec([1, 2, 3])
        >>> M = listlist2mat([[1, 0], [0, 1], [0, 0]])
        >>> project(M, x)
        Vec({0, 1, 2},{0: 1, 1: 2, 2: 0})
        >>> M = listlist2mat([[3/5, 0], [1/5, 2/3], [1/5, 1/3]])
        >>> project(M, x)
        Vec({0, 1, 2},{0: 0.96, 1: 1.8755555555555554, 2: 1.0977777777777777})
    '''
    return projected_representation(M, x) * M

## Task 11

# see documentation for image.image2display

## Task 12

