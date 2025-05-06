#include "TensorView.hpp"

#include <iostream>

using namespace tensor;

template <typename T>
void check_iterator_concepts(const std::string &name)
{
  static_assert(std::input_iterator<typename T::iterator>);
  static_assert(std::input_iterator<typename T::const_iterator>);
  static_assert(std::sentinel_for<typename T::iterator, typename T::iterator>);
  static_assert(std::sentinel_for<typename T::const_iterator, typename T::const_iterator>);
  static_assert(std::random_access_iterator<typename T::iterator>);
  static_assert(std::random_access_iterator<typename T::const_iterator>);

  std::cout << name << "::iterator and " << name << "::const_iterator satisfy iterator traits." << std::endl;
}

int main()
{
  check_iterator_concepts<Tensor<double,1>>("Tensor");
  check_iterator_concepts<TensorView<double,1>>("TensorView");
  check_iterator_concepts<FixedTensor<double,1,2>>("FixedTensor");
  check_iterator_concepts<FixedTensorView<double,3,3>>("FixedTensorView");
  check_iterator_concepts<SubView<double,2>>("SubView");

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
