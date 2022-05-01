# %%
from typing import Callable
from typing import Tuple
import math
from typing import List

Vector = List[float]

height_weight_age = [70,  # inches,
                     170,  # pounds,
                     40]  # years

grades = [95,  # exam1
          80,  # exam2
          75,  # exam3
          62]  # exam4


# %%

Vector = List[float]


def add(v: Vector, w: Vector) -> Vector:
    """Adds corresponding elements"""
    assert len(v) == len(w), "vectors must be the same length"
    return [v_i + w_i for v_i, w_i in zip(v, w)]


print(add([1, 2, 3], [4, 5, 6]))

# %%

Vector = List[float]


def subtract(v: Vector, w: Vector) -> Vector:
    """Adds corresponding elements"""
    assert len(v) == len(w), "vectors must be the same length"
    return [v_i - w_i for v_i, w_i in zip(v, w)]


print(subtract([5, 7, 9], [4, 5, 6]))


# %%

Vector = List[float]


def vector_sum(vectors: List[Vector]) -> Vector:
    """Sums all corresponding elements"""
    # Check that vectors is not empty
    assert vectors, "no vectors provided!"
    # # Check the vectors are all the same size
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different sizes!"
    # the i-th element of the result is the sum of every vector[i]
    return [sum(vector[i] for vector in vectors) for i in range(num_elements)]


print(vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]))


# %%

Vector = List[float]


def scalar_multiply(c: float, v: Vector) -> Vector:
    """Multiplies every element by c"""
    return [c * v_i for v_i in v]


print(scalar_multiply(2, [1, 2, 3]))


# %%

Vector = List[float]


def vector_mean(vectors: List[Vector]) -> Vector:
    """Computes the element-wise average"""
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))


print(vector_mean([[1, 2], [3, 4], [5, 6]]))

# %%

Vector = List[float]


def dot(v: Vector, w: Vector) -> float:
    """Computes v_1 * w_1 + ... + v_n * w_n"""
    assert len(v) == len(w), "vectors must be same length"
    return sum(v_i * w_i for v_i, w_i in zip(v, w))


print(dot([1, 2, 3], [4, 5, 6]))  # 1 * 4 + 2 * 5 + 3 * 6


# %%

Vector = List[float]


def sum_of_squares(v: Vector) -> float:
    """Returns v_1 * v_1 + ... + v_n * v_n"""
    return dot(v, v)


print(sum_of_squares([1, 2, 3]))  # 1 * 1 + 2 * 2 + 3 *


# %%

Vector = List[float]


def magnitude(v: Vector) -> float:
    """Returns the magnitude (or length) of v"""
    return math.sqrt(sum_of_squares(v))  # math.sqrt is square root function


print(magnitude([3, 4]))  # 5


# %%

Vector = List[float]


def squared_distance(v: Vector, w: Vector) -> float:
    """Computes (v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2"""
    return sum_of_squares(subtract(v, w))


def distance(v: Vector, w: Vector) -> float:
    """Computes the distance between v and w"""
    return math.sqrt(squared_distance(v, w))


print(distance([3, 4], [1, 2]))  # 2.8284271247461903


# %%

Vector = List[float]


def distance(v: Vector, w: Vector) -> float:
    return magnitude(subtract(v, w))


print(distance([3, 4], [1, 2]))  # 2.8284271247461903


# %%

Matrix = List[List[float]]

A = [[1, 2, 3],  # A has 2 rows and 3 columns
     [4, 5, 6]]

B = [[1, 2],  # B has 3 rows and 2 columns
     [3, 4],
     [5, 6]]


# %%


def shape(A: Matrix) -> Tuple[int, int]:
    """Returns (# of rows of A, # of columns of A)"""
    num_rows = len(A)
    num_cols = len(A[0])  # number of elements in first row
    return num_rows, num_cols


print(shape([[1, 2, 3], [4, 5, 6]]))  # 2 rows, 3 columns


# %%

Matrix = List[List[float]]
Vector = List[float]


def get_row(A: Matrix, i: int) -> Vector:
    """Returns the i-th row of A (as a Vector)"""
    return A[i]  # A[i] is already the ith row


print("Display the first row of A")
print(get_row([[1, 2, 3], [4, 5, 6]], 0))  # 2 rows, 3 columns
print("Display the second row of A")
print(get_row([[1, 2, 3], [4, 5, 6]], 1))  # 2 rows, 3 columns


# %%

Matrix = List[List[float]]
Vector = List[float]


def get_column(A: Matrix, j: int) -> Vector:
    """Returns the j-th column of A (as a Vector)"""
    # jth element of row A_i
    # for each row A_i
    return [A_i[j] for A_i in A]


print("Display the first column of A")
print(get_column([[1, 2, 3], [4, 5, 6]], 0))  # 2 rows, 3 columns
print("Display the second column of A")
print(get_column([[1, 2, 3], [4, 5, 6]], 1))  # 2 rows, 3 columns
print("Display the third column of A")
print(get_column([[1, 2, 3], [4, 5, 6]], 2))  # 2 rows, 3 columns

# %%

Matrix = List[List[float]]
Vector = List[float]


def make_matrix(num_rows: int,
                num_cols: int,
                entry_fn: Callable[[int, int], float]) -> Matrix:
    """
    Returns a num_rows x num_cols matrix
    whose (i,j)-th entry is entry_fn(i, j)
    """
    return [[entry_fn(i, j)             # given i, create a list
             for j in range(num_cols)]  # [entry_fn(i, 0), ... ]
            for i in range(num_rows)]   # create one list for each i


make_matrix(2, 3, lambda i, j: i * j)

# %%

Matrix = List[List[float]]
Vector = List[float]


def identity_matrix(n: int) -> Matrix:
    """Returns the n x n identity matrix"""
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)


print(identity_matrix(5))
