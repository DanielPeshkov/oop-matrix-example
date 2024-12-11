import unittest
import numpy as np
from numpy.testing import assert_allclose
from Matrix import Matrix
import os

class TestMatrix(unittest.TestCase):
    
    def setUp(self):
        self.m1 = Matrix([[1, 2], [3, 4]])
        self.m2 = Matrix([[5, 6], [7, 8]])
        self.m3 = Matrix([[9, 10, 11], [12, 13, 14]])
        self.m4 = Matrix([[1]])
        self.m5 = Matrix([[0, 0], [0, 0]])
        self.m6 = Matrix([[1, 2, 3]])
        self.m7 = Matrix([[1], [2], [3]])
        self.m8 = Matrix([[-1e6, 1e6], [1e6, -1e6]])
        self.m9 = Matrix([[1, 2], [2, 4]]) 

        # Temporary file path for saving and loading matrices
        self.test_file = 'test_matrix.npy'

    def tearDown(self):
        # Clean up the test file after each test
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_save_and_load(self):
        # Test saving and loading a basic matrix
        self.m1.save(self.test_file)
        loaded_matrix = Matrix.load(self.test_file)
        assert_allclose(self.m1.data(), loaded_matrix.data())

        # Test saving and loading a 1x1 matrix
        self.m4.save(self.test_file)
        loaded_matrix = Matrix.load(self.test_file)
        assert_allclose(self.m4.data(), loaded_matrix.data())

        # Test saving and loading a non-square matrix (2x3)
        self.m3.save(self.test_file)
        loaded_matrix = Matrix.load(self.test_file)
        assert_allclose(self.m3.data(), loaded_matrix.data())

        # Test saving and loading a singular matrix (zero matrix)
        self.m5.save(self.test_file)
        loaded_matrix = Matrix.load(self.test_file)
        assert_allclose(self.m5.data(), loaded_matrix.data())

        # Test saving and loading a matrix with extreme values
        self.m8.save(self.test_file)
        loaded_matrix = Matrix.load(self.test_file)
        assert_allclose(self.m8.data(), loaded_matrix.data())

        # Test saving and loading a large matrix (1000x1000)
        large_matrix = Matrix(np.random.rand(1000, 1000))
        large_matrix.save(self.test_file)
        loaded_matrix = Matrix.load(self.test_file)
        assert_allclose(large_matrix.data(), loaded_matrix.data())

        # Test loading from an invalid file path
        with self.assertRaises(FileNotFoundError):
            Matrix.load('invalid_path.npy')

        # Test loading from a corrupted file
        # Save a valid matrix
        self.m1.save(self.test_file)
        
        # Corrupt the file by writing random data
        with open(self.test_file, 'wb') as f:
            f.write(b'corrupted data')

        # Try to load the corrupted file
        with self.assertRaises(ValueError):
            Matrix.load(self.test_file)

        # Test that loading a matrix with incompatible shape raises an error
        matrix1 = Matrix([[1, 2], [3, 4]])
        matrix2 = Matrix([[5, 6], [7, 8]])
        
        # Save matrix1
        matrix1.save(self.test_file)
        
        # Load matrix1 and try to compare it with matrix2 (should fail)
        loaded_matrix = Matrix.load(self.test_file)
        with self.assertRaises(AssertionError):
            assert_allclose(loaded_matrix.data(), matrix2.data())

        # Test saving and loading a matrix with specific dtype (int)
        m_int = Matrix([[1, 2], [3, 4]])
        m_int.save(self.test_file)
        loaded_matrix = Matrix.load(self.test_file)
        assert_allclose(m_int.data(), loaded_matrix.data())

        # Test saving and loading an empty matrix
        empty_matrix = Matrix([[]])
        empty_matrix.save(self.test_file)
        loaded_matrix = Matrix.load(self.test_file)
        assert_allclose(empty_matrix.data(), loaded_matrix.data())

    def test_constructor(self):
        assert_allclose(self.m1.data(), np.array([[1, 2], [3, 4]]))
        assert_allclose(self.m2.data(), np.array([[5, 6], [7, 8]]))

    def test_empty_matrix(self):
        empty_matrix = Matrix([])
        assert_allclose(empty_matrix.data(), np.array([]))
        
        empty_transpose = empty_matrix.transpose()
        assert_allclose(empty_transpose.data(), np.array([]))

        with self.assertRaises(np.linalg.LinAlgError):
            empty_matrix.determinant()

    def test_str_repr(self):
        self.assertEqual(str(self.m1), '[[1 2]\n [3 4]]')
        self.assertEqual(repr(self.m1), '[[1 2]\n [3 4]]')

    def test_transpose(self):
        # Test transpose of a regular 2x2 matrix
        m1_transposed = self.m1.T()
        expected = np.array([[1, 3], [2, 4]])
        assert_allclose(m1_transposed.data(), expected)
        
        # Test transpose of a regular non-square matrix (2x3)
        m3_transposed = self.m3.transpose()
        expected_m3_transposed = np.array([[9, 12], [10, 13], [11, 14]])
        assert_allclose(m3_transposed.data(), expected_m3_transposed)

        # Test transpose of a 1x1 matrix (should be the same as original)
        m4_transposed = self.m4.transpose()
        expected_m4_transposed = np.array([[1]])
        assert_allclose(m4_transposed.data(), expected_m4_transposed)

        # Test transpose of a zero matrix (should remain a zero matrix)
        m5_transposed = self.m5.transpose()
        expected_m5_transposed = np.array([[0, 0], [0, 0]])
        assert_allclose(m5_transposed.data(), expected_m5_transposed)

        # Test transpose of a 1x3 matrix (should become a 3x1 matrix)
        m6_transposed = self.m6.transpose()
        expected_m6_transposed = np.array([[1], [2], [3]])
        assert_allclose(m6_transposed.data(), expected_m6_transposed)

        # Test transpose of a 3x1 matrix (should become a 1x3 matrix)
        m7_transposed = self.m7.transpose()
        expected_m7_transposed = np.array([[1, 2, 3]])
        assert_allclose(m7_transposed.data(), expected_m7_transposed)

        # Test transpose of a matrix with extreme values (large and small)
        m8_transposed = self.m8.transpose()
        expected_m8_transposed = np.array([[-1e6, 1e6], [1e6, -1e6]])
        assert_allclose(m8_transposed.data(), expected_m8_transposed)

        # Test transpose of an empty matrix (should remain empty)
        empty_matrix = Matrix([])
        empty_transposed = empty_matrix.transpose()
        assert_allclose(empty_transposed.data(), np.array([]))

        # Test transpose of a large matrix (100x100)
        large_matrix = Matrix(np.random.rand(100, 100))
        large_matrix_transposed = large_matrix.transpose()
        assert_allclose(large_matrix_transposed.data(), np.transpose(large_matrix.data()))

    def test_determinant(self):
        # Regular 2x2 matrix determinant
        det_m1 = self.m1.determinant()
        expected_det_m1 = np.linalg.det(np.array([[1, 2], [3, 4]]))
        assert_allclose(det_m1, expected_det_m1)
        
        # Another regular 2x2 matrix determinant
        det_m2 = self.m2.determinant()
        expected_det_m2 = np.linalg.det(np.array([[5, 6], [7, 8]]))
        assert_allclose(det_m2, expected_det_m2)

        # Non-square matrix (should raise error)
        with self.assertRaises(np.linalg.LinAlgError):
            self.m3.determinant()

        # Identity matrix (1x1)
        det_m4 = self.m4.determinant()
        expected_det_m4 = np.linalg.det(np.array([[1]]))
        assert_allclose(det_m4, expected_det_m4)

        # Zero matrix determinant (should be zero)
        det_m5 = self.m5.determinant()
        expected_det_m5 = np.linalg.det(np.array([[0, 0], [0, 0]]))
        assert_allclose(det_m5, expected_det_m5)

        # Singular matrix determinant (should be zero)
        det_m9 = self.m9.determinant()
        expected_det_m9 = np.linalg.det(np.array([[1, 2], [2, 4]]))
        assert_allclose(det_m9, expected_det_m9)

        # Extreme values matrix determinant
        det_m8 = self.m8.determinant()
        expected_det_m8 = np.linalg.det(np.array([[-1e6, 1e6], [1e6, -1e6]]))
        assert_allclose(det_m8, expected_det_m8)

        # Non-square matrix (should raise error)
        with self.assertRaises(np.linalg.LinAlgError):
            self.m6.determinant()

        # Non-square matrix (should raise error)
        with self.assertRaises(np.linalg.LinAlgError):
            self.m7.determinant()

        # Large matrix (100x100) determinant
        large_matrix = Matrix(np.random.rand(100, 100))
        det_large_matrix = large_matrix.determinant()
        expected_det_large_matrix = np.linalg.det(large_matrix.data())
        assert_allclose(det_large_matrix, expected_det_large_matrix)

        # Nearly singular matrix (determinant close to zero)
        nearly_singular_matrix = Matrix([[1, 1e-10], [1, 1]])
        det_nearly_singular = nearly_singular_matrix.determinant()
        expected_det_nearly_singular = np.linalg.det(np.array([[1, 1e-10], [1, 1]]))
        assert_allclose(det_nearly_singular, expected_det_nearly_singular)

    def test_inverse(self):
        # Test invertible 2x2 matrix
        m1_inv = self.m1.inverse()
        expected_m1_inv = np.linalg.inv(np.array([[1, 2], [3, 4]]))
        assert_allclose(m1_inv.data(), expected_m1_inv)

        # Test another invertible 2x2 matrix
        m2_inv = self.m2.inverse()
        expected_m2_inv = np.linalg.inv(np.array([[5, 6], [7, 8]]))
        assert_allclose(m2_inv.data(), expected_m2_inv)

        # Test non-square matrix (should raise error)
        with self.assertRaises(np.linalg.LinAlgError):
            self.m3.inverse()

        # Test identity matrix (inverse should be the same as the identity matrix)
        m4_inv = self.m4.inverse()
        expected_m4_inv = np.linalg.inv(np.array([[1]]))  # Inverse of 1x1 identity matrix is itself
        assert_allclose(m4_inv.data(), expected_m4_inv)

        # Test zero matrix (cannot be inverted, should raise error)
        with self.assertRaises(np.linalg.LinAlgError):
            self.m5.inverse()

        # Test non-square 1x3 matrix (should raise error)
        with self.assertRaises(np.linalg.LinAlgError):
            self.m6.inverse()

        # Test non-square 3x1 matrix (should raise error)
        with self.assertRaises(np.linalg.LinAlgError):
            self.m7.inverse()

        # Test matrix with extreme values (with 0 determinant)
        with self.assertRaises(np.linalg.LinAlgError):
            self.m8.inverse()

        # Test singular matrix (should raise error)
        with self.assertRaises(np.linalg.LinAlgError):
            self.m9.inverse()

        # Test large matrix (100x100, should be invertible if it has full rank)
        large_matrix = Matrix(np.random.rand(100, 100))
        large_matrix_inv = large_matrix.inverse()
        expected_large_matrix_inv = np.linalg.inv(large_matrix.data())
        assert_allclose(large_matrix_inv.data(), expected_large_matrix_inv)

    def test_eigenvalue(self):
        # Test for m1 (2x2 matrix)
        eigenvalues_m1, _ = self.m1.eigenvalue()
        expected_m1_eigenvalues = np.linalg.eig(np.array([[1, 2], [3, 4]]))[0]
        assert_allclose(eigenvalues_m1, expected_m1_eigenvalues)

        # Test for m2 (2x2 matrix)
        eigenvalues_m2, _ = self.m2.eigenvalue()
        expected_m2_eigenvalues = np.linalg.eig(np.array([[5, 6], [7, 8]]))[0]
        assert_allclose(eigenvalues_m2, expected_m2_eigenvalues)

        # Test for m3 (2x3 non-square matrix, should raise an error)
        with self.assertRaises(np.linalg.LinAlgError):
            self.m3.eigenvalue()

        # Test for m4 (1x1 identity matrix)
        eigenvalues_m4, _ = self.m4.eigenvalue()
        expected_m4_eigenvalues = np.linalg.eig(np.array([[1]]))[0]  # Eigenvalue of identity matrix should be 1
        assert_allclose(eigenvalues_m4, expected_m4_eigenvalues)

        # Test for m5 (zero matrix)
        eigenvalues_m5, _ = self.m5.eigenvalue()
        expected_m5_eigenvalues = np.linalg.eig(np.array([[0, 0], [0, 0]]))[0]  # Eigenvalue of zero matrix should be 0
        assert_allclose(eigenvalues_m5, expected_m5_eigenvalues)

        # Test for m6 (1x3 non-square matrix, should raise an error)
        with self.assertRaises(np.linalg.LinAlgError):
            self.m6.eigenvalue()

        # Test for m7 (3x1 non-square matrix, should raise an error)
        with self.assertRaises(np.linalg.LinAlgError):
            self.m7.eigenvalue()

        # Test for m8 (extreme values matrix)
        eigenvalues_m8, _ = self.m8.eigenvalue()
        expected_m8_eigenvalues = np.linalg.eig(np.array([[-1e6, 1e6], [1e6, -1e6]]))[0]
        assert_allclose(eigenvalues_m8, expected_m8_eigenvalues)

        # Test for m9 (singular matrix)
        eigenvalues_m9, _ = self.m9.eigenvalue()
        expected_m9_eigenvalues = np.linalg.eig(np.array([[1, 2], [2, 4]]))[0]  # One eigenvalue should be 0
        assert_allclose(eigenvalues_m9, expected_m9_eigenvalues)

    def test_diag(self):
        # Test for m1 (2x2 matrix, k=0, main diagonal)
        result_m1 = self.m1.diag(k=0).data()
        expected_m1 = np.array([1, 4])
        assert_allclose(result_m1, expected_m1)

        # Test for m1 (2x2 matrix, k=1, upper diagonal)
        result_m1_k1 = self.m1.diag(k=1).data()
        expected_m1_k1 = np.array([2])  
        assert_allclose(result_m1_k1, expected_m1_k1)

        # Test for m1 (2x2 matrix, k=-1, lower diagonal)
        result_m1_k_neg1 = self.m1.diag(k=-1).data()
        expected_m1_k_neg1 = np.array([3]) 
        assert_allclose(result_m1_k_neg1, expected_m1_k_neg1)

        # Test for m2 (2x2 matrix, k=0)
        result_m2 = self.m2.diag(k=0).data()
        expected_m2 = np.array([5, 8])
        assert_allclose(result_m2, expected_m2)

        # Test for m3 (2x3 matrix, k=0)
        result_m3 = self.m3.diag(k=0).data()
        expected_m3 = np.array([9, 13]) 
        assert_allclose(result_m3, expected_m3)

        # Test for m4 (1x1 matrix, k=0)
        result_m4 = self.m4.diag(k=0).data()
        expected_m4 = np.array([1]) 
        assert_allclose(result_m4, expected_m4)

        # Test for m5 (zero matrix, k=0)
        result_m5 = self.m5.diag(k=0).data()
        expected_m5 = np.array([0, 0])
        assert_allclose(result_m5, expected_m5)

        # Test for m6 (1x3 matrix, k=0, non-square)
        result_m6 = self.m6.diag(k=0).data()
        expected_m6 = np.array([1])
        assert_allclose(result_m6, expected_m6)

        # Test for m7 (3x1 matrix, k=0, non-square)
        result_m7 = self.m7.diag(k=0).data()
        expected_m7 = np.array([1]) 
        assert_allclose(result_m7, expected_m7)

        # Test for m8 (Extreme values matrix, k=0)
        result_m8 = self.m8.diag(k=0).data()
        expected_m8 = np.array([-1e6, -1e6]) 
        assert_allclose(result_m8, expected_m8)

        # Test for m9 (singular matrix, k=0)
        result_m9 = self.m9.diag(k=0).data()
        expected_m9 = np.array([1, 4]) 
        assert_allclose(result_m9, expected_m9)

    def test_add(self):
        # Adding a scalar to a matrix
        result = self.m1 + 10
        expected = np.array([[11, 12], [13, 14]])
        assert_allclose(result.data(), expected)

        # Adding a scalar to a matrix (e.g., 10 + m1)
        result = 10 + self.m1
        expected = np.array([[11, 12], [13, 14]])  
        assert_allclose(result.data(), expected)

        # Adding two matrices with the same shape
        result = self.m1 + self.m2
        expected = np.array([[6, 8], [10, 12]])
        assert_allclose(result.data(), expected)

        # Test adding matrices of incompatible shapes
        with self.assertRaises(ValueError):
            self.m1 + self.m3  

        # Test adding a singular matrix (zero matrix)
        result = self.m1 + self.m5
        expected = np.array([[1, 2], [3, 4]]) 
        assert_allclose(result.data(), expected)

        # Test adding singular matrix (linearly dependent rows)
        result = self.m1 + self.m9
        expected = np.array([[2, 4], [5, 8]]) 
        assert_allclose(result.data(), expected)

        # Test adding extreme values
        result = self.m1 + self.m8
        expected = np.array([[-999999, 1000002], [1000003, -999996]])
        assert_allclose(result.data(), expected)

        # Test adding a matrix with an invalid data type (e.g., a string)
        with self.assertRaises(TypeError):
            self.m1 + "invalid_data"

        # Test adding a matrix with invalid data (e.g., None)
        with self.assertRaises(TypeError):
            self.m1 + None  

        # Test adding a scalar to matrices of different shapes
        result = self.m6 + 5 
        expected = np.array([[6, 7, 8]])
        assert_allclose(result.data(), expected)

        # Test adding an unsupported type (e.g., a string matrix)
        with self.assertRaises(TypeError):
            self.m1 + Matrix([["a", "b"], ["c", "d"]]) 

        # Test adding a non-numeric matrix
        with self.assertRaises(TypeError):
            self.m1 + Matrix([["one", "two"], ["three", "four"]]) 

    def test_sub(self):
        # Subtract a scalar from a matrix (e.g., 10 - m1)
        result = 10 - self.m1
        expected = np.array([[9, 8], [7, 6]])  
        assert_allclose(result.data(), expected)

        # Subtract two matrices with the same shape (2x2)
        result = self.m2 - self.m1
        expected = np.array([[4, 4], [4, 4]]) 
        assert_allclose(result.data(), expected)

        # Test subtracting matrices of incompatible shapes (e.g., 2x2 and 2x3)
        with self.assertRaises(ValueError):
            self.m2 - self.m3 

        # Subtracting from a singular matrix (e.g., zero matrix)
        result = self.m5 - self.m2 
        expected = np.array([[-5, -6], [-7, -8]]) 
        assert_allclose(result.data(), expected)

        # Subtracting from a singular matrix (e.g., m9 with linearly dependent rows)
        result = self.m9 - self.m2
        expected = np.array([[-4, -4], [-5, -4]]) 
        assert_allclose(result.data(), expected)

        # Subtracting from a matrix with extreme values (e.g., large positive and negative values)
        result = self.m8 - self.m2
        expected = np.array([[-1000005, 999994], [999993, -1000008]]) 
        assert_allclose(result.data(), expected)

        # Test subtracting a matrix with an invalid data type (e.g., string)
        with self.assertRaises(TypeError):
            self.m1 - "invalid_data" 

        # Test subtracting a matrix with an unsupported matrix type (e.g., string matrix)
        with self.assertRaises(TypeError):
            self.m1 - Matrix([["a", "b"], ["c", "d"]]) 

        # Test subtracting a matrix with non-numeric data
        with self.assertRaises(TypeError):
            self.m1 - Matrix([["one", "two"], ["three", "four"]]) 

        # Test subtracting matrices of different shapes (e.g., 1x3 and 1x2 matrices)
        with self.assertRaises(ValueError):
            self.m6 - self.m5 

        # Test subtracting non-square matrices (e.g., 1x3 matrix)
        result = self.m6 - self.m7
        expected = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]) 
        assert_allclose(result.data(), expected)

    def test_mul(self):
        # Multiply a matrix by a scalar
        result = 2 * self.m1
        expected = np.array([[2, 4], [6, 8]]) 
        assert_allclose(result.data(), expected)

        # Multiply two matrices with the same shape (2x2)
        result = self.m1 * self.m2
        expected = np.array([[5, 12], [21, 32]]) 
        assert_allclose(result.data(), expected)

        # Test multiplying matrices with incompatible shapes (2x2 and 2x3)
        with self.assertRaises(ValueError):
            self.m1 * self.m3 

        # Test multiplying with a singular matrix (e.g., zero matrix)
        result = self.m5 * self.m2  
        expected = np.array([[0, 0], [0, 0]])  
        assert_allclose(result.data(), expected)

        # Test multiplying with a singular matrix with linearly dependent rows
        result = self.m9 * self.m2
        expected = np.array([[5, 12], [14, 32]])  
        assert_allclose(result.data(), expected)

        # Test multiplying with extreme values (e.g., large numbers)
        result = self.m8 * self.m2
        expected = np.array([[-5e6, 6e6], [7e6, -8e6]]) 
        assert_allclose(result.data(), expected)

        # Test multiplying with invalid data type (string)
        with self.assertRaises(TypeError):
            self.m1 * "invalid_data" 

        # Test multiplying with a matrix that has non-numeric data
        with self.assertRaises(TypeError):
            self.m1 * Matrix([["one", "two"], ["three", "four"]])

        # Test multiplying non-square matrices (e.g., 1x3 and 3x1)
        result = self.m6 * self.m7 
        expected = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]]) 
        assert_allclose(result.data(), expected)

        # Test multiplying with a 1x1 matrix
        result = self.m4 * self.m1
        expected = np.array([[1, 2], [3, 4]]) 
        assert_allclose(result.data(), expected)

    def test_matmul(self):
        # Test matmul against scalar (invalid dimensions)
        with self.assertRaises(ValueError):
            self.m1 @ 2

        # Test multiplying two matrices with the same shape (2x2)
        result = self.m1 @ self.m2
        expected = np.array([[19, 22], [43, 50]])  # Matrix multiplication result
        assert_allclose(result.data(), expected)

        # Test multiplying matrices with different but compatible shapes (2x2 and 2x3)
        result = self.m1 @ self.m3  
        expected = np.array([[33, 36, 39], [75, 82, 89]])
        assert_allclose(result.data(), expected)

        # Test multiplying with a singular matrix (zero matrix)
        result = self.m5 @ self.m2 
        expected = np.array([[0, 0], [0, 0]]) 
        assert_allclose(result.data(), expected)

        # Test multiplying with a singular matrix with linearly dependent rows
        result = self.m9 @ self.m2
        expected = np.array([[19, 22], [38, 44]]) 
        assert_allclose(result.data(), expected)

        # Test multiplying with extreme values
        result = self.m8 @ self.m2
        expected = np.array([[ 2000000, 2000000.], [-2000000, -2000000.]])
        assert_allclose(result.data(), expected)

        # Test multiplying with invalid data type (string)
        with self.assertRaises(TypeError):
            self.m1 @ "invalid_data"

        # Test multiplying with a matrix that has non-numeric data
        with self.assertRaises(TypeError):
            self.m1 @ Matrix([["one", "two"], ["three", "four"]]) 

        # Test multiplying non-square matrices (e.g., 1x3 and 3x1)
        result = self.m6 @ self.m7 
        expected = np.array([[14]]) 
        assert_allclose(result.data(), expected)

        # Test multiplying with a 1x1 matrix
        with self.assertRaises(ValueError):
            self.m4 @ self.m1

    def test_div(self):
        # Test dividing a matrix by a scalar
        result = self.m1 / 2
        expected = np.array([[0.5, 1], [1.5, 2]]) 
        assert_allclose(result.data(), expected)

        # Test dividing by a 1x1 matrix (acts like scalar division)
        result = self.m1 / self.m4
        expected = np.array([[1, 2], [3, 4]]) 
        assert_allclose(result.data(), expected)

        # Test dividing by a zero matrix (produces np.inf and warning)
        with self.assertWarns(RuntimeWarning):
            result = self.m1 / self.m5 
            expected = np.array([[np.inf, np.inf], [np.inf, np.inf]]) 
            assert_allclose(result.data(), expected)
        
        # Test dividing matrices with incompatible shapes (2x2 by 2x3)
        with self.assertRaises(ValueError):
            self.m1 / self.m3 

        # Test division by a singular matrix (e.g., rows are linearly dependent)
        result = self.m9 / self.m2
        expected = np.array([[0.2, 0.33333333], [0.28571429, 0.5]]) 
        assert_allclose(result.data(), expected)

        # Test dividing matrices with extreme values
        result = self.m8 / 1e6
        expected = np.array([[-1, 1], [1, -1]]) 
        assert_allclose(result.data(), expected)

        # Test dividing by a non-numeric value
        with self.assertRaises(TypeError):
            self.m1 / "string_value"

        # Test dividing by a matrix with non-numeric values 
        with self.assertRaises(TypeError):
            self.m1 / Matrix([["one", "two"], ["three", "four"]]) 

        # Test division by zero scalar (produces np.inf and warning)
        with self.assertWarns(RuntimeWarning):
            result = self.m1 / 0
            expected = np.array([[np.inf, np.inf], [np.inf, np.inf]]) 
            assert_allclose(result.data(), expected)

        # Test division of non-square matrices (1x3 by 3x1)
        result = self.m6 / self.m7
        expected = np.array([[1, 2, 3], [0.5, 1, 1.5], [0.33333333333, 0.66666666667, 1]])
        assert_allclose(result.data(), expected)

    def test_floordiv(self):
        # Test dividing a matrix by a scalar using floor division
        result = self.m1 // 2
        expected = np.array([[0, 1], [1, 2]]) 
        assert_allclose(result.data(), expected)

        # Test dividing by a 1x1 matrix
        result = self.m1 // self.m4
        expected = np.array([[1, 2], [3, 4]]) 
        assert_allclose(result.data(), expected)

        # Test dividing by a zero matrix (produces 0 and warning)
        with self.assertWarns(RuntimeWarning):
            result = self.m1 // self.m5
            expected = np.array([[0, 0], [0, 0]]) 
            assert_allclose(result.data(), expected)

        # Test dividing matrices with incompatible shapes (2x2 by 2x3)
        with self.assertRaises(ValueError):
            self.m1 // self.m3 

        # Test floor division by a singular matrix
        result = self.m9 // self.m2
        expected = np.array([[0, 0], [0, 0]]) 
        assert_allclose(result.data(), expected)

        # Test dividing matrices with extreme values
        result = self.m8 // 1e6
        expected = np.array([[-1, 1], [1, -1]]) 
        assert_allclose(result.data(), expected)

        # Test dividing by a non-numeric value 
        with self.assertRaises(TypeError):
            self.m1 // "string_value" 

        # Test dividing by a matrix with non-numeric values 
        with self.assertRaises(TypeError):
            self.m1 // Matrix([["one", "two"], ["three", "four"]]) 

        # Test floor division by zero scalar (produces 0 and warning)
        with self.assertWarns(RuntimeWarning):
            result = self.m1 // 0
            expected = np.array([[0, 0], [0, 0]]) 
            assert_allclose(result.data(), expected)

        # Test floor division of non-square matrices (1x3 by 3x1)
        result = self.m6 // self.m7
        expected = np.array([[1, 2, 3], [0, 1, 1], [0, 0, 1]]) 
        assert_allclose(result.data(), expected)

    def test_identity(self):
        # Test identity matrix for a 2x2 case
        result = Matrix.identity(2)
        expected = np.array([[1., 0.], [0., 1.]])
        assert_allclose(result.data(), expected)

        # Test identity matrix for a 1x1 case
        result = Matrix.identity(1)
        expected = np.array([[1.]])
        assert_allclose(result.data(), expected)

        # Test identity matrix for a 0x0 case (should return an empty matrix)
        result = Matrix.identity(0)
        expected = np.empty((0, 0))
        assert_allclose(result.data(), expected)

        # Test identity matrix with a specific dtype (e.g., float32)
        result = Matrix.identity(3, dtype=np.float32)
        expected = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=np.float32)
        assert_allclose(result.data(), expected)

        # Test invalid dtype, should raise error
        with self.assertRaises(TypeError):
            Matrix.identity(3, dtype="invalid")

        # Test identity matrix for a large size (e.g., 1000x1000)
        result = Matrix.identity(1000)
        expected = np.identity(1000)
        assert_allclose(result.data(), expected)

        # Test identity matrix for negative dimension
        with self.assertRaises(ValueError):
            Matrix.identity(-2)

        # Test identity matrix for non-integer dimension
        with self.assertRaises(TypeError):
            Matrix.identity(2.5)

        # Test identity matrix with extreme dtype values (e.g., high precision float)
        result = Matrix.identity(2, dtype=np.float64)
        expected = np.array([[1., 0.], [0., 1.]], dtype=np.float64)
        assert_allclose(result.data(), expected, rtol=1e-9)

        # Test identity matrix for large size (e.g., 10000x10000)
        result = Matrix.identity(10000)
        expected = np.identity(10000)
        assert_allclose(result.data(), expected)

    def test_zeros(self):
        # Test zeros for a 2x2 matrix
        result = Matrix.zeros((2, 2))
        expected = np.zeros((2, 2))
        assert_allclose(result.data(), expected)

        # Test zeros for a 1x1 matrix
        result = Matrix.zeros((1, 1))
        expected = np.zeros((1, 1))
        assert_allclose(result.data(), expected)

        # Test zeros for a 3x3 matrix
        result = Matrix.zeros((3, 3))
        expected = np.zeros((3, 3))
        assert_allclose(result.data(), expected)

        # Test zeros for a non-square matrix (3x2)
        result = Matrix.zeros((3, 2))
        expected = np.zeros((3, 2))
        assert_allclose(result.data(), expected)

        # Test zeros for a matrix with a specific dtype
        result = Matrix.zeros((2, 2), dtype=int)
        expected = np.zeros((2, 2), dtype=int)
        assert_allclose(result.data(), expected)

        # Test zeros for an invalid dtype
        with self.assertRaises(TypeError):
            Matrix.zeros((2, 2), dtype="invalid")

        # Test zeros for an empty shape
        result = Matrix.zeros((0, 0))
        expected = np.zeros((0, 0))
        assert_allclose(result.data(), expected)

        # Test zeros for negative dimensions
        with self.assertRaises(ValueError):
            Matrix.zeros((-2, 2))

        # Test zeros for one zero dimension
        result = Matrix.zeros((0, 2))
        expected = np.zeros((0, 2))
        assert_allclose(result.data(), expected)

        # Test zeros for non-integer dimensions
        with self.assertRaises(TypeError):
            Matrix.zeros((2.5, 3))

        # Test zeros for a large matrix (1000x1000)
        result = Matrix.zeros((1000, 1000))
        expected = np.zeros((1000, 1000))
        assert_allclose(result.data(), expected)

    def test_ones(self):
        # Test ones for a 2x2 matrix
        result = Matrix.ones((2, 2))
        expected = np.ones((2, 2))
        assert_allclose(result.data(), expected)

        # Test ones for a 1x1 matrix
        result = Matrix.ones((1, 1))
        expected = np.ones((1, 1))
        assert_allclose(result.data(), expected)

        # Test ones for a 3x3 matrix
        result = Matrix.ones((3, 3))
        expected = np.ones((3, 3))
        assert_allclose(result.data(), expected)

        # Test ones for a non-square matrix (3x2)
        result = Matrix.ones((3, 2))
        expected = np.ones((3, 2))
        assert_allclose(result.data(), expected)

        # Test ones for a matrix with a specific dtype
        result = Matrix.ones((2, 2), dtype=int)
        expected = np.ones((2, 2), dtype=int)
        assert_allclose(result.data(), expected)

        # Test ones for an invalid dtype
        with self.assertRaises(TypeError):
            Matrix.ones((2, 2), dtype="invalid")

        # Test ones for an empty shape
        result = Matrix.ones((0, 0))
        expected = np.ones((0, 0))
        assert_allclose(result.data(), expected)

        # Test ones for negative dimensions
        with self.assertRaises(ValueError):
            Matrix.ones((-2, 2))

        # Test ones for one zero dimension
        result = Matrix.ones((0, 2))
        expected = np.ones((0, 2))
        assert_allclose(result.data(), expected)

        # Test ones for non-integer dimensions
        with self.assertRaises(TypeError):
            Matrix.ones((2.5, 3))

        # Test ones for a large matrix (1000x1000)
        result = Matrix.ones((1000, 1000))
        expected = np.ones((1000, 1000))
        assert_allclose(result.data(), expected)

    def test_eye(self):
        # Test identity matrix creation (3x3)
        result = Matrix.eye(3)
        expected = np.eye(3)
        assert_allclose(result.data(), expected)

        # Test identity matrix creation for a square matrix (4x4)
        result = Matrix.eye(4)
        expected = np.eye(4)
        assert_allclose(result.data(), expected)

        # Test identity matrix creation for a non-square matrix (2x3)
        result = Matrix.eye(2, 3)
        expected = np.eye(2, 3)
        assert_allclose(result.data(), expected)

        # Test identity matrix with a positive k (upper diagonal)
        result = Matrix.eye(3, k=1)
        expected = np.eye(3, k=1)
        assert_allclose(result.data(), expected)

        # Test identity matrix with a negative k (lower diagonal)
        result = Matrix.eye(3, k=-1)
        expected = np.eye(3, k=-1)
        assert_allclose(result.data(), expected)

        # Test identity matrix creation with a specific dtype (e.g., int)
        result = Matrix.eye(3, dtype=int)
        expected = np.eye(3, dtype=int)
        assert_allclose(result.data(), expected)

        # Test identity matrix creation with an invalid dtype (should raise TypeError)
        with self.assertRaises(TypeError):
            Matrix.eye(3, dtype="invalid")

        # Test identity matrix creation with zero rows and columns (should return empty matrix)
        result = Matrix.eye(0, 0)
        expected = np.eye(0, 0)
        assert_allclose(result.data(), expected)

        # Test identity matrix creation with negative rows (should raise ValueError)
        with self.assertRaises(ValueError):
            Matrix.eye(-3)

        # Test identity matrix creation with negative columns (should raise ValueError)
        with self.assertRaises(ValueError):
            Matrix.eye(3, -3)

        # Test identity matrix creation with zero rows (should return an empty matrix)
        result = Matrix.eye(0, 3)
        expected = np.eye(0, 3)
        assert_allclose(result.data(), expected)

        # Test identity matrix creation with zero columns (should return an empty matrix)
        result = Matrix.eye(3, 0)
        expected = np.eye(3, 0)
        assert_allclose(result.data(), expected)

        # Test identity matrix creation for a very large matrix (1000x1000)
        result = Matrix.eye(1000)
        expected = np.eye(1000)
        assert_allclose(result.data(), expected)

        # Test identity matrix creation with invalid shape (should raise TypeError)
        with self.assertRaises(TypeError):
            Matrix.eye("3", 3)

    def test_full(self):
        # Test the creation of a matrix filled with a constant value (2x2)
        result = Matrix.full((2, 2), 5)
        expected = np.full((2, 2), 5)
        assert_allclose(result.data(), expected)

        # Test the creation of a matrix filled with a single value (1x1)
        result = Matrix.full((1, 1), 7)
        expected = np.full((1, 1), 7)
        assert_allclose(result.data(), expected)

        # Test the creation of a non-square matrix filled with a constant (2x3)
        result = Matrix.full((2, 3), 0)
        expected = np.full((2, 3), 0)
        assert_allclose(result.data(), expected)

        # Test the creation of a large matrix (1000x1000) filled with a constant
        result = Matrix.full((1000, 1000), 1)
        expected = np.full((1000, 1000), 1)
        assert_allclose(result.data(), expected)

        # Test the creation of a matrix with a specified dtype (int)
        result = Matrix.full((2, 2), 5, dtype=int)
        expected = np.full((2, 2), 5, dtype=int)
        assert_allclose(result.data(), expected)

        # Test invalid dtype input (should raise TypeError)
        with self.assertRaises(TypeError):
            Matrix.full((2, 2), 5, dtype="invalid")

        # Test the creation of a matrix filled with zeros
        result = Matrix.full((3, 3), 0)
        expected = np.full((3, 3), 0)
        assert_allclose(result.data(), expected)

        # Test the creation of a matrix filled with negative values
        result = Matrix.full((2, 2), -3)
        expected = np.full((2, 2), -3)
        assert_allclose(result.data(), expected)

        # Test the creation of a matrix filled with extreme values (e.g., 1e6)
        result = Matrix.full((2, 2), 1e6)
        expected = np.full((2, 2), 1e6)
        assert_allclose(result.data(), expected)

        # Test invalid shape input (should raise TypeError)
        with self.assertRaises(TypeError):
            Matrix.full("2, 2", 5)

        # Test zero-dimensional matrix (empty matrix)
        result = Matrix.full((0, 0), 0)
        expected = np.full((0, 0), 0)
        assert_allclose(result.data(), expected)

        # Test the case where rows are negative (should raise ValueError)
        with self.assertRaises(ValueError):
            Matrix.full((-2, 3), 5)

        # Test the case where columns are negative (should raise ValueError)
        with self.assertRaises(ValueError):
            Matrix.full((2, -3), 5)

        # Test with very large values as fill_value (e.g., 1e12)
        result = Matrix.full((3, 3), 1e12)
        expected = np.full((3, 3), 1e12)
        assert_allclose(result.data(), expected)

if __name__ == "__main__":
    unittest.main()
