# TensorView

A header-only C++ library for multi-dimensional array (tensor) manipulation with zero-cost abstractions. TensorView provides both owning (`Tensor`) and non-owning (`TensorView`) containers with support for various memory layouts and operations.

## Features

- **Header-only library**: Simply include `TensorView.hpp` to get started.
- **Zero-cost abstractions**: Views and iterators compile down to efficient code.
- **Flexible memory layouts**: Support for both C-order (row-major) and Fortran-order (column-major).
- **Multiple memory spaces**: Host and device (CUDA) memory support.
- **Type-safe indexing**: Multi-dimensional indexing with compile-time dimension checks.
- **STL-compatible**: Iterators work with standard algorithms.
- **Subviews and slicing**: Efficient views into tensor subsets without copying data.
- **Reshape operations**: Change tensor dimensions without reallocating.

## Requirements

- C++20 or later
- CMake 3.10 or later
- (Optional) CUDA for device memory support

## Installation

### Using CMake

```bash
git clone https://github.com/arotem3/TensorView.git
cd TensorView
mkdir build && cd build
cmake ..
make
sudo make install
```

### Including in Your Project

```cmake
find_package(TensorView REQUIRED)
target_link_libraries(your_target PRIVATE TensorView::TensorView)
```

Or simply copy the `TensorView/` directory and include the header:

```cpp
#include "TensorView.hpp"
```

## Quick Start

### Basic Usage

```cpp
#include "TensorView.hpp"
using namespace tensor;

// Create a 3x4 matrix (2D tensor)
Matrix<double> mat(3, 4);

// Access elements
mat(1, 2) = 3.14;
double val = mat(1, 2);

// Linear indexing also supported
mat[5] = 2.71;

// Iterate over all elements
for (auto& elem : mat) {
    elem = 1.0;
}
```

### Creating Tensors

```cpp
// Create tensors of various dimensions
Vector<double> vec(100);              // 1D vector
Matrix<double> mat(10, 20);           // 2D matrix
Cube<double> cube(5, 10, 15);         // 3D cube
Tensor<double, 4> tensor4d(2, 3, 4, 5); // 4D tensor

auto a = makeLike<int>(cube); // t has the same shape at cube but int values
```

### Working with Views

TensorView provides non-owning views that reference existing data without copying:

```cpp
// Raw array
double data[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

// Create a view as a 3x4 matrix (Fortran order by default)
TensorView<double, 2> view(data, 3, 4);

// Access through the view
view(1, 2) = 42.0;  // Modifies data[7]

// Views can be reshaped
auto reshaped = reshape(data, 2, 6);  // Now a 2x6 matrix
```

### Memory Layout: C-order vs Fortran-order

```cpp
double data[6] = {1, 2, 3, 4, 5, 6};

// Fortran order (column-major): elements stored column by column
FTensorView<double, 2> f_view(data, 2, 3);
// Layout: | 1 3 5 |
//         | 2 4 6 |
// f_view(0, 0) = data[0] = 1
// f_view(1, 0) = data[1] = 2
// f_view(0, 1) = data[2] = 3

// C order (row-major): elements stored row by row
CTensorView<double, 2> c_view(data, 2, 3);
// Layout: | 1 2 3 |
//         | 4 5 6 |
// c_view(0, 0) = data[0] = 1
// c_view(0, 1) = data[1] = 2
// c_view(1, 0) = data[3] = 4
```

### Slicing and Subviews

Create efficient views into tensor subsets:

```cpp
Tensor<double, 4> tensor(5, 10, 8, 6);

// Extract a subview using fancy indexing
// All() - all indices in that dimension
// Span(start, end) - indices from start to end (exclusive)
auto sub = tensor(All(), 2, Span(0, 4), Span(1, 5));
// Result: 3D tensor of shape [5, 4, 4]

// The subview references the original data
sub(0, 0, 0) = 99.0;  // Modifies tensor(0, 2, 0, 1)
```

### Reshaping

```cpp
// Reshape in-place (for owning tensors)
Tensor<double, 2> tensor(6, 10);
tensor.reshape(10, 6);  // Now 10x6

// Create reshaped view (for existing data)
double data[24];
auto view1 = reshape(data, 4, 6);   // 4x6 view
auto view2 = reshape(data, 2, 3, 4); // 2x3x4 view

// Views maintain reference to original data
view1(1, 2) = 5.0;
// data has been modified
```

### Type Aliases

Convenient aliases for common tensor types:

```cpp
using namespace tensor;

// 1D, 2D, 3D tensors (owning)
Vector<float> v(100);
Matrix<double> m(10, 20);
Cube<int> c(5, 10, 15);

// Non-owning views
VectorView<float> vv(data, 100);
MatrixView<double> mv(data, 10, 20);
CubeView<int> cv(data, 5, 10, 15);
```

### Iterators and STL Algorithms

TensorView provides STL-compatible iterators:

```cpp
#include <algorithm>
#include <numeric>

Vector<double> x(100);

// Fill with values
std::iota(x.begin(), x.end(), 0.0);

// Use STL algorithms
auto max_elem = std::max_element(x.begin(), x.end());
std::sort(x.begin(), x.end());

// Range-based for loops
for (const auto& val : x) {
    std::cout << val << " ";
}
```

## API Overview

### Tensor (Owning Container)

- `Tensor<T, NumDims>` - Fortran-order tensor
- `CTensor<T, NumDims>` - C-order tensor
- Methods:
  - `shape(i)` - Get size of dimension i
  - `size()` - Total number of elements
  - `data()` - Pointer to underlying data
  - `operator()(i, j, ...)` - Multi-dimensional indexing
  - `operator[i]` - Linear indexing
  - `reshape(dims...)` - Change dimensions in-place
  - `at(indices...)` - Create subview with slicing

### TensorView (Non-Owning View)

- `TensorView<T, NumDims>` - Fortran-order view
- `CTensorView<T, NumDims>` - C-order view
- Same interface as `Tensor` but doesn't own data

### PersistentViews (Partially-Owning Views)

`PersistentView` is a special view type that extends the lifetime of the underlying data through a `shared_ptr`. Unlike regular views (`TensorView`) which are purely non-owning, persistent views maintain a reference to the data source, ensuring it remains valid for the view's lifetime.

**When are PersistentViews created?**

- **Fancy indexing/subviews**: When you slice a tensor using `at()` or the `operator()` with `Span()` or `All()`
- **Explicit views**: When calling `view()` on an owning tensor

```cpp
Tensor<double, 3> tensor(10, 20, 30);

// Fancy indexing produces a PersistentView
auto sub = tensor(Span(0, 5), All(), 10);  // PersistentView<...>

// Explicit view also produces a PersistentView
auto view = tensor.view();  // PersistentView<...>

// The PersistentView keeps the original tensor data alive
// even if 'tensor' goes out of scope

// `persistent` is valid even though `t` is local to the lambda.
auto persistent = []() {
    auto t = makeTensor(2, 3, 4);
    return t(All(), 2, Span(1, 2));
}();
```

**Benefits:**

- **Safety**: Prevents dangling references when returning subviews from functions
- **Convenience**: Subviews can outlive their parent tensor without manual lifetime management
- **Flexibility**: Can be passed around like owning containers while maintaining view semantics
- **Interface**: `PersistentView` provides the same interface as `TensorView`.

### Utility Functions

- `reshape(data, dims...)` - Create reshaped view of data
- `makeTensor<T>(dims...)` - Convenient tensor creation
- `Span(begin, end, stride=1)` - Index range for slicing
- `All()` - All indices in a dimension

## Building Tests

```bash
cd build
cmake ..
make
ctest
```

## Examples

### Matrix Operations

```cpp
Matrix<double> A(3, 3);
Matrix<double> B(3, 3);

// Initialize matrices
for (index_t i = 0; i < 3; ++i) {
    for (index_t j = 0; j < 3; ++j) {
        A(i, j) = i + j;
        B(i, j) = i - j;
    }
}

A(All(), 0) = B(1, All()); // Copy into view
```

### Working with External Libraries

```cpp
// Interface with libraries expecting raw pointers
Matrix<double> mat(100, 100);

// Pass to BLAS/LAPACK or other C libraries
external_function(mat.data(), mat.shape(0), mat.shape(1));

// Wrap external data without copying
double* external_data = get_data_from_library();
TensorView<double, 2> view(external_data, rows, cols);
```

## Acknowledgments

TensorView is designed for high-performance computing applications requiring flexible multi-dimensional array manipulation with minimal overhead.
