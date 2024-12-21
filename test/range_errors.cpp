#include "TensorView.hpp"

#include <iostream>

using namespace tensor;

int main()
{
  double data[6] = {1, 2, 3, 4, 5, 6};
  TensorView<double, 2> tensor_view(data, 2, 3);

#ifdef TENSOR_DEBUG
  int fails = 3;
  
  try
  {
    tensor_view(2, 3);
  }
  catch (const std::out_of_range &e)
  {
    std::cout << "Caught exception as expected: " << e.what() << std::endl;
    fails--;
  }

  try
  {
    tensor_view[6];
  }
  catch (const std::out_of_range &e)
  {
    std::cout << "Caught exception as expected: " << e.what() << std::endl;
    fails--;
  }

  try
  {
    tensor_view(all(), span(1, 9));
  }
  catch(const std::out_of_range &e)
  {
    std::cout << "Caught exception as expected: " << e.what() << std::endl;
    fails--;
  }
  

  if (fails)
  {
    std::cout << "Out of range errors test failed!" << std::endl;
  }
  else
  {
    std::cout << "Out of range errors test passed!" << std::endl;
  }

  return fails;
#else
  return 0;
#endif
}
