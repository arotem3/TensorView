# TensorView
`TensorView` is a single header template library defining the class `TensorView` which is used to structure
contiguous arrays into tensor-like objects (like numpy's `ndarray`) whose rank (dimension) is known at compile time.

The `TensorView` is a trivial class and only stores the pointer to the start of the array and the shape of the tensor.

`TensorView` objects can also be used in CUDA code by enabling the `USE_CUDA` option in cmake. If not using cmake define the macro `TENSOR_USE_CUDA` before including `TensorView.hpp`.
Since the `TensorView` object is trivial, it can wrap device pointers and passed by value to a CUDA `__global__` function.

By default, indexing into a `TensorView` is unsafe. To enable bound checks when compiling with cmake either:

* configure cmake with `-D CMAKE_BUILD_TYPE=Debug` or
* enable `-D TENSOR_DEBUG=ON` option.

If not using cmake, define the macro `TENSOR_DEBUG` before including `TensorView.hpp`.

## Example

The following is an example using the TensorView class:
```c++
#include "TensorView.hpp"

using namespace tensor;

int main()
{
    double x[6];

    auto matrix = reshape(x, 2, 3);
    // decltype(matrix) == matrix_view<double> == TensorView<2, double>

    for (index_t j = 0; j < 3; ++j)
    {
        for (index_t i = 0; i < 2; ++i)
        {
            matrix(i, j) = i*j;
        }
    }


    return 0;
}
```

To compile this example:
```
c++ -std=c++17 main.cpp -I path/to/TensorView
```