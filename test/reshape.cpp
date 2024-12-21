#include "TensorView.hpp"

#include <iostream>

using namespace tensor;

int main()
{
  double data[6] = {1, 2, 3, 4, 5, 6};
  TensorView<double, 2> tensor_view(data, 2, 3);
  FixedTensorView<double, 2, 3> fixed_tensor_view(data);
  Tensor<double, 2> tensor = make_tensor<double>(2, 3);
  FixedTensor<double, 2, 3> fixed_tensor;

  std::cout << "Reshaping tensor_view inplace..." << std::endl;
  tensor_view.reshape(3, 2);

  std::cout << "Reshaping tensor inplace..." << std::endl;
  tensor.reshape(3, 2);

  std::cout << "Reshaping pointer into TensorView..." << std::endl;
  auto reshaped_data = reshape(data, 1, 6);

  std::cout << "Reshaping tensor_view into TensorView..." << std::endl;
  auto reshaped_tensor_view = reshape(tensor_view, 1, 6);

  std::cout << "Reshaping fixed_tensor_view into TensorView..." << std::endl;
  auto reshaped_fixed_tensor_view = reshape(fixed_tensor_view, 1, 6);

  std::cout << "Reshaping tensor into TensorView..." << std::endl;
  auto reshaped_tensor = reshape(tensor, 1, 6);

  std::cout << "Reshaping fixed_tensor into TensorView..." << std::endl;
  auto reshaped_fixed_tensor = reshape(fixed_tensor, 1, 6);

  std::cout << "Reshape test passed!" << std::endl;

  return 0;
}
