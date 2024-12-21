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

  int fails = 0;

  int position = 0;
  for (auto it = tensor_view.begin(); it != tensor_view.end(); it++)
  {
    fails += *it != data[position];
    position++;
  }

  position = 0;
  for (auto it = fixed_tensor_view.begin(); it != fixed_tensor_view.end(); it++)
  {
    fails += *it != data[position];
    position++;
  }

  if (fails)
  {
    std::cout << "Iterator test failed!" << std::endl;
  }
  else
  {
    std::cout << "Iterator test passed!" << std::endl;
  }

  return fails;
}
