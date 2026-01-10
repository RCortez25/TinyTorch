import numpy as np

class Tensor:
    """Tensor - the foundation of machine learning computation.

    This class provides the core data structure for all ML operations:
    - data: The actual numerical values (NumPy array)
    - shape: Dimensions of the tensor
    - size: Total number of elements
    - dtype: Data type (float32)

    All arithmetic, matrix, and shape operations are built on this foundation.
    """
    def __init__(self, data):
        """Create a new tensor from data."""
        self.data = np.array(data, dtype=np.float32)
        self.shape = self.data.shape
        self.size = self.data.size
        self.dtype = self.data.dtype

    def __repr__(self):
        "String representation for debugging"
        return f"Tensor(data={self.data}, shape={self.shape})"

    def __str__(self):
        "String representation"
        return f"Tensor({self.data})"

    def numpy(self):
        "Return the NumPy array"
        return self.data

    def memory_footprint(self):
        """Calculate exact memory usage in bytes.

        Systems Concept: Understanding memory footprint is fundamental to ML systems.
        Before running any operation, engineers should know how much memory it requires.

        Returns:
            int: Memory usage in bytes (e.g., 1000x1000 float32 = 4MB)
        """
        return self.data.nbytes

    def __add__(self, other):
        """Add two tensors element-wise with broadcasting support."""
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        return self.data + other

    def __sub__(self, other):
        """Subtract two tensors element-wise."""

        if isinstance(other, Tensor):
            return Tensor(self.data - other.data)
        return self.data - other

    def __mul__(self, other):
        """Multiply two tensors element-wise (NOT matrix multiplication)."""
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data)
        return self.data * other

    def __truediv__(self, other):
        """Divide two tensors element-wise."""
        if isinstance(other, Tensor):
            return Tensor(self.data / other.data)
        return Tensor(self.data / other)

    def matmul(self, other):
        """Matrix multiplication of two tensors."""
        if not isinstance(other, Tensor):
            raise TypeError("Both elements must be tensors")
        if other.data.ndim == 0:
            return Tensor(self * other)
        if other.data.ndim == 1:
            return Tensor(np.matmul(self.data, other.data))
        if not self.shape[-1] == other.shape[-2]:
            raise ValueError(f"Inner dimensions must match. {self.shape[-1]} ≠ {other.shape[-2]}")
        if self.data.shape == (2,2):
            lst = []
            for i in range(2):
                for j in range(2):
                    result = np.dot(self.data[i,:],other.data[:,j])
                    lst.append(result)
            return Tensor(np.array(lst).reshape(2,2))
        return Tensor(np.matmul(self.data, other.data))

    def __matmul__(self, other):
        """Enable @ operator for matrix multiplication."""
        return self.matmul(other)

    def __getitem__(self, key):
        """Enable indexing and slicing operations on Tensors."""
        return Tensor(np.array(self.data[key]))

    def reshape(self, *shape):
        """Reshape tensor to new dimensions.
        - For -1: unknown_dim = self.size // known_size in the tuple
        """
        if isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        if shape[-1] == -1:
            unknown_dim = self.size // shape[0]
            shape = (shape[0],unknown_dim)
        if self.data.size != np.prod(shape):
            raise ValueError(f"Total elements must match. {self.data.size} ≠ {np.prod(shape)}")
        return Tensor(self.data.reshape(shape))

    def transpose(self, dim0=None, dim1=None):
        """Transpose tensor dimensions."""
        axes = list(range(len(self.shape)))
        if len(axes) == 1:
            return self
        if dim0 is None and dim1 is None:
            axes[-2], axes[-1] = axes[-1], axes[-2]
            return Tensor(np.transpose(self.data, axes))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return Tensor(np.transpose(self.data, axes))

    def sum(self, axis=None, keepdims=False):
        """Sum tensor along specified axis."""
        return Tensor(np.sum(self.data, axis=axis, keepdims=keepdims))

    def mean(self, axis=None, keepdims=False):
        """Compute mean of tensor along specified axis."""
        result = np.mean(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result)

    def max(self, axis=None, keepdims=False):
        """Find maximum values along specified axis."""
        result = np.max(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result)