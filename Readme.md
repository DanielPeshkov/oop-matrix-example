# Matrix Class Documentation

This `Matrix` class provides a wrapper around NumPy arrays, enabling commonly used matrix operations. The class supports both scalar and matrix operations, and includes methods for saving and loading matrices to/from files.

## Table of Contents

- [Constructor](#constructor)
- [Instance Methods](#instance-methods)
  - [String Representation Methods](#string-representation-methods)
  - [Matrix Operations Methods](#matrix-operations-methods)
  - [Matrix Properties](#matrix-properties)
  - [I/O Methods](#io-methods)
- [Static Methods](#static-methods)
- [Arithmetic Operations](#arithmetic-operations)

## Constructor

### `__init__(self, data)`
Initializes a `Matrix` object with the provided data, converting it to a NumPy array.

**Parameters**:
- `data`: A list, tuple, or any array-like object that can be converted to a NumPy array.

---

## Instance Methods

### String Representation Methods

#### `__str__(self)`
Returns a human-readable string representation of the matrix.

**Returns**:
- A string representing the matrix in a readable format.

#### `__repr__(self)`
Returns a more detailed string representation of the matrix, suitable for debugging.

**Returns**:
- A string representation of the matrix.

### Matrix Operations Methods

#### `data(self)`
Returns the matrix data as a NumPy array.

**Returns**:
- A NumPy array representing the matrix.

#### `transpose(self)`
Returns the transpose of the matrix as a new `Matrix` object.

**Returns**:
- A new `Matrix` object containing the transpose of the current matrix.

#### `T(self)`
Alias for `transpose`. Returns the transpose of the matrix.

**Returns**:
- A new `Matrix` object containing the transpose of the current matrix.

#### `determinant(self)`
Returns the determinant of the matrix.

**Returns**:
- A scalar value representing the determinant of the matrix.

#### `det(self)`
Alias for `determinant`. Returns the determinant of the matrix.

**Returns**:
- A scalar value representing the determinant of the matrix.

#### `inverse(self)`
Returns the inverse of the matrix as a new `Matrix` object.

**Returns**:
- A new `Matrix` object containing the inverse of the matrix.

#### `inv(self)`
Alias for `inverse`. Returns the inverse of the matrix.

**Returns**:
- A new `Matrix` object containing the inverse of the matrix.

#### `eigenvalue(self)`
Returns the eigenvalues and eigenvectors of the matrix.

**Returns**:
- A tuple containing the eigenvalues and eigenvectors of the matrix.

#### `eig(self)`
Alias for `eigenvalue`. Returns the eigenvalues and eigenvectors of the matrix.

**Returns**:
- A tuple containing the eigenvalues and eigenvectors of the matrix.

#### `diag(self, k=0)`
Returns a new matrix with the diagonal elements of the current matrix. Optionally, a diagonal offset can be specified.

**Parameters**:
- `k`: The diagonal offset (default is `0`, the main diagonal).

**Returns**:
- A new `Matrix` object containing the diagonal elements.

### I/O Methods

#### `save(self, path: str)`
Saves the matrix data to a `.npy` file.

**Parameters**:
- `path`: The file path where the matrix will be saved.

#### `load(path: str)`
Loads matrix data from a `.npy` file and returns a new `Matrix` object.

**Parameters**:
- `path`: The file path from which the matrix will be loaded.

**Returns**:
- A new `Matrix` object containing the data loaded from the specified file.

---

## Static Methods

### `identity(n, dtype=None)`
Creates an identity matrix of size `n x n`.

**Parameters**:
- `n`: The size of the matrix (number of rows and columns).
- `dtype`: Optional data type for the matrix elements (default is `float`).

**Returns**:
- A new `Matrix` object representing the identity matrix.

### `zeros(shape, dtype=float)`
Creates a matrix filled with zeros.

**Parameters**:
- `shape`: A tuple representing the shape of the matrix (rows, columns).
- `dtype`: Optional data type for the matrix elements (default is `float`).

**Returns**:
- A new `Matrix` object filled with zeros.

### `ones(shape, dtype=float)`
Creates a matrix filled with ones.

**Parameters**:
- `shape`: A tuple representing the shape of the matrix (rows, columns).
- `dtype`: Optional data type for the matrix elements (default is `float`).

**Returns**:
- A new `Matrix` object filled with ones.

### `eye(rows, cols=None, k=0, dtype=float)`
Creates a matrix with ones on the diagonal and zeros elsewhere.

**Parameters**:
- `rows`: The number of rows.
- `cols`: The number of columns (optional; defaults to `rows`).
- `k`: The diagonal offset (default is `0`, the main diagonal).
- `dtype`: Optional data type for the matrix elements (default is `float`).

**Returns**:
- A new `Matrix` object with ones on the diagonal.

### `full(shape, fill_value, dtype=None)`
Creates a matrix filled with a specified value.

**Parameters**:
- `shape`: A tuple representing the shape of the matrix (rows, columns).
- `fill_value`: The value to fill the matrix with.
- `dtype`: Optional data type for the matrix elements.

**Returns**:
- A new `Matrix` object filled with the specified value.

---

## Arithmetic Operations

The following arithmetic operations are supported for both scalar and matrix operations. These methods use Python's magic methods (e.g., `__add__`, `__sub__`) to enable the use of operators (`+`, `-`, `*`, `@`, `/`, `//`).

### Left-hand operations (Matrix-Object)

- **Addition**: `__add__(self, other)` (Matrix + Scalar or Matrix)
- **Subtraction**: `__sub__(self, other)` (Matrix - Scalar or Matrix)
- **Multiplication**: `__mul__(self, other)` (Matrix * Scalar or Matrix)
- **Matrix Multiplication**: `__matmul__(self, other)` (Matrix @ Scalar or Matrix)
- **Division**: `__truediv__(self, other)` (Matrix / Scalar or Matrix)
- **Floor Division**: `__floordiv__(self, other)` (Matrix // Scalar or Matrix)

### Right-hand operations (Object-Matrix)

These methods allow matrices to participate in operations with non-matrix objects.

- **Addition**: `__radd__(self, other)` (Scalar + Matrix)
- **Subtraction**: `__rsub__(self, other)` (Scalar - Matrix)
- **Multiplication**: `__rmul__(self, other)` (Scalar * Matrix)
- **Division**: `__rtruediv__(self, other)` (Scalar / Matrix)
- **Floor Division**: `__rfloordiv__(self, other)` (Scalar // Matrix)

---

## Example Usage

```python
# Creating matrices
m1 = Matrix([[1, 2], [3, 4]])
m2 = Matrix([[5, 6], [7, 8]])

# Basic Operations
m3 = m1 + m2  # Matrix addition
m4 = m1 - m2  # Matrix subtraction
m5 = m1 * 2   # Scalar multiplication
m6 = m1 @ m2  # Matrix multiplication

# Matrix Properties
det = m1.determinant()  # Determinant
inv = m1.inverse()      # Inverse

# Static Methods
m7 = Matrix.zeros((3, 3))  # 3x3 matrix of zeros
m8 = Matrix.identity(4)    # 4x4 identity matrix
