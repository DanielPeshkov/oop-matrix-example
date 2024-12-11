import numpy as np
from functools import singledispatchmethod

class Matrix:
    def __init__(self, data):
        self.__data = np.array(data)

    def __str__(self):
        return np.array2string(self.__data)
    
    def __repr__(self):
        return np.array2string(self.__data)
    
    def data(self):
        return self.__data
    
    def transpose(self):
        return Matrix(np.transpose(self.__data))
    
    def T(self):
        return Matrix(np.transpose(self.__data))
    
    def determinant(self):
        return np.linalg.det(self.__data)
    
    def det(self):
        return np.linalg.det(self.__data)
    
    def inverse(self):
        return Matrix(np.linalg.inv(self.__data))
    
    def inv(self):
        return Matrix(np.linalg.inv(self.__data))
    
    def eigenvalue(self):
        return np.linalg.eig(self.__data)
    
    def eig(self):
        return np.linalg.eig(self.__data)
    
    def diag(self, k=0):
        return Matrix(np.diag(self.__data, k))
    
    def save(self, path: str):
        np.save(path, self.__data)

    def load(path: str):
        return Matrix(np.load(path))

    # Left hand operations
    @singledispatchmethod
    def __add__(self, other):
        return Matrix(self.__data + other)

    @singledispatchmethod
    def __sub__(self, other):
        return Matrix(self.__data - other)
    
    @singledispatchmethod
    def __mul__(self, other):
        return Matrix(self.__data * other)
    
    @singledispatchmethod
    def __matmul__(self, other):
        return Matrix(self.__data @ other)
    
    @singledispatchmethod
    def __truediv__(self, other):
        return Matrix(self.__data / other)
    
    @singledispatchmethod
    def __floordiv__(self, other):
        return Matrix(self.__data // other)
    
    # Right hand operations
    @singledispatchmethod
    def __radd__(self, other):
        return Matrix(other + self.__data)

    @singledispatchmethod
    def __rsub__(self, other):
        return Matrix(other - self.__data)
    
    @singledispatchmethod
    def __rmul__(self, other):
        return Matrix(other * self.__data)
    
    @singledispatchmethod
    def __rtruediv__(self, other):
        return Matrix(other / self.__data)
    
    @singledispatchmethod
    def __rfloordiv__(self, other):
        return Matrix(other // self.__data)
    
    def identity(n, dtype=None):
        return Matrix(np.identity(n, dtype))
    
    def zeros(shape, dtype=float):
        return Matrix(np.zeros(shape, dtype))
    
    def ones(shape, dtype=float):
        return Matrix(np.ones(shape, dtype))
    
    def eye(rows, cols=None, k=0, dtype=float):
        return Matrix(np.eye(rows, cols, k, dtype))
    
    def full(shape, fill_value, dtype=None):
        return Matrix(np.full(shape, fill_value, dtype))

# Matrix to Matrix ops
@Matrix.__add__.register
def _(self, other: Matrix):
    return Matrix(self.data() + other.data())

@Matrix.__sub__.register
def _(self, other: Matrix):
    return Matrix(self.data() - other.data())

@Matrix.__mul__.register
def _(self, other: Matrix):
    return Matrix(self.data() * other.data())

@Matrix.__matmul__.register
def _(self, other: Matrix):
    return Matrix(self.data() @ other.data())

@Matrix.__truediv__.register
def _(self, other: Matrix):
    return Matrix(self.data() / other.data())

@Matrix.__floordiv__.register
def _(self, other: Matrix):
    return Matrix(self.data() // other.data())