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

  std::cout << "Implicit conversion from Tensor to TensorView..." << std::endl;
  Tensor<float, 2> t;
  TensorView<float, 2> view1(t);         // OK
  TensorView<const float, 2> view2(t);   // OK

  const Tensor<float, 2> ct;
  TensorView<const float, 2> view3(ct);  // OK
  // TensorView<float, 2> view4(ct);     // Error: ct is const
  // TensorView<float, 2> view_to_temp(Tensor<float, 2>{}); // Error: dangling reference

  std::cout << "Initialization test passed!" << std::endl;
  return 0;
}
