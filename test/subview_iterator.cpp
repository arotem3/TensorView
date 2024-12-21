#include "TensorView.hpp"

#include <iostream>

using namespace tensor;

int main()
{
  double data[1000];
  for (int i = 0; i < 1000; i++)
  {
    data[i] = static_cast<double>(rand()) / RAND_MAX;
  }

  TensorView<double, 2> tensor_view(data, 10, 100);
  FixedTensorView<double, 10, 100> fixed_tensor_view(data);

  auto tensor_view_subview = tensor_view.at(4, all{});
  auto fixed_tensor_view_subview = fixed_tensor_view.at(4, all{});

  const int stride = 10;
  const int offset = 4;

  int fails = 0;

  int pos = 0;
  for (auto val : tensor_view_subview)
  {
    fails += val != data[offset + pos * stride];
    pos++;
  }

  pos = 0;
  for (auto val : fixed_tensor_view_subview)
  {
    fails += val != data[offset + pos * stride];
    pos++;
  }

  if (fails)
  {
    std::cout << "Subview iterator test failed!" << std::endl;
  }
  else
  {
    std::cout << "Subview iterator test passed!" << std::endl;
  }

  return fails;
}
