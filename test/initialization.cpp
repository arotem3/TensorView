#include "TensorView.hpp"

#include <iostream>
#include <cassert>

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
  Tensor<float, 2> t(2, 3);

  TensorView<float, 2> view1(t); // OK
  view1 = make_view(t);          // OK

  TensorView<const float, 2> view2(t); // OK
  view2 = make_view(t);                // OK

  const Tensor<float, 2> ct(2, 3);
  TensorView<const float, 2> view3(ct); // OK
  view3 = make_view(ct);                // OK

  // TensorView<float, 2> view4(ct);     // Error: ct is const
  // TensorView<float, 2> view5 = make_view(ct); // Error: ct is const
  // TensorView<float, 2> view6(Tensor<float, 2>(1,1)); // Error: dangling reference
  // TensorView<float, 2> view7 = make_view(Tensor<float, 2>{}); // Error: view7 is dangling reference

  std::cout << "Initialize with lower dimension..." << std::endl;
  Tensor<double, 3> tensor3(2, 3);
  assert(tensor3.shape(0) == 2);
  assert(tensor3.shape(1) == 3);
  assert(tensor3.shape(2) == 1);

  TensorView<float, 3> view3d_of_2d(t);
  assert(view3d_of_2d.shape(0) == 2);
  assert(view3d_of_2d.shape(1) == 3);
  assert(view3d_of_2d.shape(2) == 1);

  std::cout << "Initialization test passed!" << std::endl;
  return 0;
}
