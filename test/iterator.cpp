#include "test.hpp"
using namespace tensor;

template <typename T>
static int test_iterator_traits(const std::string &name)
{
   int n_fails = 0;

   if (std::ranges::random_access_range<const T>)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ] ") << name << "::const_iterator is a random access iterator."
                << std::endl;
   }
   else
   {
      std::cout << "\t" << ColorText::red("[ ✗ ] ") << name << "::const_iterator is NOT a random access iterator."
                << std::endl;
      n_fails++;
   }

   if (std::ranges::output_range<T, typename T::value_type>)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ] ") << name << "::iterator is an output iterator." << std::endl;
   }
   else
   {
      std::cout << "\t" << ColorText::red("[ ✗ ] ") << name << "::iterator is NOT an output iterator." << std::endl;
      n_fails++;
   }

   if constexpr (details::TensorTraits<T>::contiguous())
   {
      if (std::ranges::contiguous_range<const T>)
      {
         std::cout << "\t" << ColorText::green("[ ✓ ] ") << name
                   << "::const_iterator is a contiguous iterator as expected." << std::endl;
      }
      else
      {
         std::cout << "\t" << ColorText::red("[ ✗ ] ") << name
                   << "::const_iterator is NOT a contiguous iterator but it should be." << std::endl;
         n_fails++;
      }
   }

   return n_fails;
}

static int test_tensorview_iterator()
{
   int n_fails = 0;

   int data[500];
   for (int i = 0; i < 500; i++)
      data[i] = rand();

   auto tensor_view = reshape(data, 5, 10, 2, 5);

   int position = 0;
   for (auto v : tensor_view)
   {
      if (v != data[position])
      {
         n_fails++;
         break;
      }
      position++;
   }

   if (n_fails == 0)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " TensorView iterator works correctly." << std::endl;
   }
   else
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " TensorView iterator failed!" << std::endl;
   }

   return n_fails;
}

static int test_tensor_iterator()
{
   int n_fails = 0;
   Tensor<int, 4> tensor(5, 10, 2, 5);

   for (index_t i = 0; i < tensor.size(); i++)
   {
      tensor.data()[i] = rand();
   }

   int position = 0;
   for (auto v : tensor)
   {
      if (v != tensor.data()[position])
      {
         n_fails++;
         break;
      }
      position++;
   }

   if (n_fails == 0)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Tensor iterator works correctly." << std::endl;
   }
   else
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Tensor iterator failed!" << std::endl;
   }

   return n_fails;
}

int main()
{
   int n_fails = 0;

   n_fails += test_iterator_traits<TensorView<double, 3>>("TensorView<double, 3>");
   n_fails += test_iterator_traits<Tensor<double, 4>>("FixedTensorView<double, 4>");
   n_fails += test_iterator_traits<PView<int, 2, LinearOrder::C, MemorySpace::Host>>("PView<int, 2>");
   n_fails += test_iterator_traits<StaticTensor<float, 2, 3>>("StaticTensor<float, 2, 3>");
   n_fails += test_iterator_traits<StaticView<double, 1, 4>>("StaticView<double, 1, 4>");

   using StridedPView = details::PersistentView<details::StridedShape<3>, int, MemorySpace::Host, false>;
   n_fails += test_iterator_traits<StridedPView>("StridedPView<int, 3>");

   using StridedRawView = details::RawView<details::StridedShape<2>, float, MemorySpace::Host>;
   n_fails += test_iterator_traits<StridedRawView>("StridedRawView<float, 2>");

   n_fails += test_tensorview_iterator();
   n_fails += test_tensor_iterator();

   if (n_fails == 0)
   {
      std::cout << ColorText::green("iterator.cpp: All tests passed!") << std::endl;
   }
   else
   {
      std::cout << ColorText::red(std::format("iterator.cpp: {} tests failed!", n_fails)) << std::endl;
   }
}
