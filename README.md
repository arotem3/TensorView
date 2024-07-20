# TensorView
`TensorView` is a single header template library defining several classes which are used to structure
contiguous arrays into tensor-like objects (like numpy's `ndarray`) whose rank (aka dimension or order) is known at compile time.

## `class TensorView`

The `TensorView` is a trivial class and only stores the pointer to the start of the array and the shape of the tensor.

`TensorView` objects can also be used in CUDA code by enabling the `USE_CUDA` option in cmake. If not using cmake define the macro `TENSOR_USE_CUDA` before including `TensorView.hpp`.
Since the `TensorView` object is trivial, it can wrap device pointers and passed by value to a CUDA `__global__` function.

By default, indexing into a `TensorView` is unsafe. To enable bound checks when compiling with cmake either:

* configure cmake with `-D CMAKE_BUILD_TYPE=Debug` or
* enable `-D TENSOR_DEBUG=ON` option.

If not using cmake, define the macro `TENSOR_DEBUG` before including `TensorView.hpp`.

### Example

The following is an example using the `TensorView` class:
```c++
#include "TensorView.hpp"

using namespace tensor;

int main()
{
    double x[6];

    auto matrix = reshape(x, 2, 3);
    // or:
    // TensorView<double, 2> matrix(x, 2, 3);

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

## `class FixedTensorView`

The `FixedTensorView` is much like `TensorView` except that the shape of the tensor is known at compile time allowing for some optimization.

### Example
The following is an example using the `FixedTensorView` class:
```c++
#include "TensorView.hpp"

using namespace tensor;

int main()
{
    double x[6];

    FixedTensorView<double, 2, 3> matrix(x);

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

## `class Tensor`

The `Tensor` is used similarly to the `TensorView` with the exception that the `Tensor` dynamically allocates and manages its own memory internally (with a `std::vector`).

### Example
The following is an example using the `Tensor` class:
```c++
#include "TensorView.hpp"

using namespace tensor;

int main()
{
    auto matrix = make_tensor<double>(2, 3);
    // or:
    // Tensor<double, 2> matrix(2, 3);

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

## `class FixedTensor`
The `FixedTensor` is a tensor whose shape is known at compile time (similar to `FixedTensorView`) and manages its own memory internally. Because the shape is known at compile time, the memory is stack allocated (the memory is managed by `std::array`). Therefore, this class is intended for smaller tensors.

### Example
The following is an example using the `FixedTensor` class:
```c++
#include "TensorView.hpp"

using namespace tensor;

int main()
{
    FixedTensor<double, 2, 3> matrix;

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