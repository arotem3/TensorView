#include "TensorView.hpp"

#include <iostream>

using namespace tensor;

int main()
{
  double data[6] = {1, 2, 3, 4, 5, 6};

  std::cout << "Initializing TensorView object..." << std::endl;
  TensorView<double, 2> tensor_view(data, 2, 3);

  std::cout << "Initializing FixedTensorView object..." << std::endl;
  FixedTensorView<double, 2, 3> fixed_tensor_view(data);

  std::cout << "Initializing Tensor object..." << std::endl;
  Tensor<double, 2> tensor = make_tensor<double>(2, 3);

  std::cout << "Initializing FixedTensor object..." << std::endl;
  FixedTensor<double, 2, 3> fixed_tensor;

  std::cout << "Initialization test passed!" << std::endl;
  return 0;
}
