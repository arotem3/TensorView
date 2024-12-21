#include "TensorView.hpp"

#include <iostream>

using namespace tensor;

int main()
{
  double data[6] = {1, 2, 3, 4, 5, 6};
  TensorView<double, 2> tensor_view(data, 2, 3);
  FixedTensorView<double, 2, 3> fixed_tensor_view(data);

  int fails = 0;
  for (int i = 0; i < 6; i++)
  {
    std::cout << "tensor_view[" << i << "] = " << tensor_view[i] << " ?= " << data[i] << std::endl;
    fails += tensor_view[i] != data[i];
  }

  for (int i = 0; i < 6; i++)
  {
    std::cout << "fixed_tensor_view[" << i << "] = " << fixed_tensor_view[i] << " ?= " << data[i] << std::endl;
    fails += fixed_tensor_view[i] != data[i];
  }

  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      std::cout << "tensor_view(" << i << ", " << j << ") = " << tensor_view(i, j) << " ?= " << data[i + 2 * j] << std::endl;
      fails += tensor_view(i, j) != data[i + 2 * j];
    }
  }

  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      std::cout << "fixed_tensor_view(" << i << ", " << j << ") = " << fixed_tensor_view(i, j) << " ?= " << data[i + 2 * j] << std::endl;
      fails += fixed_tensor_view(i, j) != data[i + 2 * j];
    }
  }

  if (fails)
  {
    std::cout << "Element access test failed!" << std::endl;
  }
  else
  {
    std::cout << "Element access test passed!" << std::endl;
  }

  return fails;
}
