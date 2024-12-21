#include "TensorView.hpp"

#include <iostream>

using namespace tensor;

int main()
{
  double data[500];
  for (int i = 0; i < 500; i++)
  {
    data[i] = static_cast<double>(rand()) / RAND_MAX;
  }

  TensorView<double, 4> tensor_view(data, 5, 10, 2, 5);
  FixedTensorView<double, 5, 10, 2, 5> fixed_tensor_view(data);
  Tensor<double, 4> tensor = make_tensor<double>(5, 10, 2, 5);
  FixedTensor<double, 5, 10, 2, 5> fixed_tensor;

  for (double &val : tensor)
    val = static_cast<double>(rand()) / RAND_MAX;
  for (double &val : fixed_tensor)
    val = static_cast<double>(rand()) / RAND_MAX;

  auto tensor_subview = tensor.at(all{}, 2, span(0, 1), span(2, 4));
  auto fixed_tensor_subview = fixed_tensor.at(all{}, 2, span(0, 1), span(2, 4));
  auto tensor_view_subview = tensor_view.at(all{}, 2, span(0, 1), span(2, 4));
  auto fixed_tensor_view_subview = fixed_tensor_view.at(all{}, 2, span(0, 1), span(2, 4));

  int fails = 0;

  for (int i = 0; i < 5; i++)
  {
    for (int j = 0; j < 1; ++j)
    {
      for (int k = 0; k < 2; ++k)
      {
        std::cout << "tensor_subview.at(" << i << ", " << j << ", " << k << ") = " << tensor_subview.at(i, j, k) << " ?= " << tensor.at(i, 2, j, 2 + k) << std::endl;
        fails += tensor_subview.at(i, j, k) != tensor.at(i, 2, j, 2 + k);
      }
    }
  }

  for (int i = 0; i < 5; i++)
  {
    for (int j = 0; j < 1; ++j)
    {
      for (int k = 0; k < 2; ++k)
      {
        std::cout << "fixed_tensor_subview.at(" << i << ", " << j << ", " << k << ") = " << fixed_tensor_subview.at(i, j, k) << " ?= " << fixed_tensor.at(i, 2, j, 2 + k) << std::endl;
        fails += fixed_tensor_subview.at(i, j, k) != fixed_tensor.at(i, 2, j, 2 + k);
      }
    }
  }

  for (int i = 0; i < 5; i++)
  {
    for (int j = 0; j < 1; ++j)
    {
      for (int k = 0; k < 2; ++k)
      {
        std::cout << "tensor_view_subview.at(" << i << ", " << j << ", " << k << ") = " << tensor_view_subview.at(i, j, k) << " ?= " << tensor_view.at(i, 2, j, 2 + k) << std::endl;
        fails += tensor_view_subview.at(i, j, k) != tensor_view.at(i, 2, j, 2 + k);
      }
    }
  }

  for (int i = 0; i < 5; i++)
  {
    for (int j = 0; j < 1; ++j)
    {
      for (int k = 0; k < 2; ++k)
      {
        std::cout << "fixed_tensor_view_subview.at(" << i << ", " << j << ", " << k << ") = " << fixed_tensor_view_subview.at(i, j, k) << " ?= " << fixed_tensor_view.at(i, 2, j, 2 + k) << std::endl;
        fails += fixed_tensor_view_subview.at(i, j, k) != fixed_tensor_view.at(i, 2, j, 2 + k);
      }
    }
  }

  if (fails)
  {
    std::cout << "Subview test failed!" << std::endl;
  }
  else
  {
    std::cout << "Subview test passed!" << std::endl;
  }

  return fails;
}
