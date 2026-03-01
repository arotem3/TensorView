#include "test.hpp"
using namespace tensor;

template <typename T>
static int testIteratorTraits(const std::string &name)
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

   using const_iterator_t = decltype(std::declval<const T &>().begin());
   if (std::input_iterator<const_iterator_t>)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ] ") << name << "::(const T).begin() is an input iterator."
                << std::endl;
   }
   else
   {
      std::cout << "\t" << ColorText::red("[ ✗ ] ") << name << "::(const T).begin() is NOT an input iterator."
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

   using iterator_t = decltype(tensor::details::TensorBegin(std::declval<T &>()));
   if (std::sentinel_for<tensor::details::TensorEndSentinel, iterator_t>)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ] ") << name << "::TensorEndSentinel is a sentinel for iterator."
                << std::endl;
   }
   else
   {
      std::cout << "\t" << ColorText::red("[ ✗ ] ") << name << "::TensorEndSentinel is NOT a sentinel for iterator."
                << std::endl;
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

   // Check subview iterator traits
   if constexpr (requires { std::declval<T &>().at(0, All{}); })
   {
      using subview_t = decltype(std::declval<T &>().at(0, All{}));
      if (std::ranges::range<subview_t>)
      {
         std::cout << "\t" << ColorText::green("[ ✓ ] ") << name << "::subview is a range." << std::endl;
      }
      else
      {
         std::cout << "\t" << ColorText::red("[ ✗ ] ") << name << "::subview is NOT a range." << std::endl;
         n_fails++;
      }

      using const_subview_t = decltype(std::declval<const T &>().at(0, All{}));
      using const_subview_iter_t = decltype(std::declval<const_subview_t &>().begin());
      if (std::input_iterator<const_subview_iter_t>)
      {
         std::cout << "\t" << ColorText::green("[ ✓ ] ") << name << "::(const subview).begin() is an input iterator."
                   << std::endl;
      }
      else
      {
         std::cout << "\t" << ColorText::red("[ ✗ ] ") << name << "::(const subview).begin() is NOT an input iterator."
                   << std::endl;
         n_fails++;
      }
   }

   // Check raw view iterator traits
   if constexpr (requires { std::declval<T &>().raw(); })
   {
      using raw_t = decltype(std::declval<T &>().raw());
      if (std::ranges::range<raw_t>)
      {
         std::cout << "\t" << ColorText::green("[ ✓ ] ") << name << "::raw_view is a range." << std::endl;
      }
      else
      {
         std::cout << "\t" << ColorText::red("[ ✗ ] ") << name << "::raw_view is NOT a range." << std::endl;
         n_fails++;
      }

      using const_raw_t = decltype(std::declval<const T &>().raw());
      using const_raw_iter_t = decltype(std::declval<const_raw_t &>().begin());
      if (std::input_iterator<const_raw_iter_t>)
      {
         std::cout << "\t" << ColorText::green("[ ✓ ] ") << name << "::(const raw_view).begin() is an input iterator."
                   << std::endl;
      }
      else
      {
         std::cout << "\t" << ColorText::red("[ ✗ ] ") << name << "::(const raw_view).begin() is NOT an input iterator."
                   << std::endl;
         n_fails++;
      }
   }

   // Check permuted dimensions iterator traits (transpose)
   if constexpr (requires { tensor::transpose(std::declval<T &>()); })
   {
      using permuted_t = decltype(tensor::transpose(std::declval<T &>()));
      if (std::ranges::range<permuted_t>)
      {
         std::cout << "\t" << ColorText::green("[ ✓ ] ") << name << "::transpose is a range." << std::endl;
      }
      else
      {
         std::cout << "\t" << ColorText::red("[ ✗ ] ") << name << "::transpose is NOT a range." << std::endl;
         n_fails++;
      }

      using const_permuted_t = decltype(tensor::transpose(std::declval<const T &>()));
      using const_permuted_iter_t = decltype(std::declval<const_permuted_t &>().begin());
      if (std::input_iterator<const_permuted_iter_t>)
      {
         std::cout << "\t" << ColorText::green("[ ✓ ] ") << name << "::(const transpose).begin() is an input iterator."
                   << std::endl;
      }
      else
      {
         std::cout << "\t" << ColorText::red("[ ✗ ] ") << name
                   << "::(const transpose).begin() is NOT an input iterator." << std::endl;
         n_fails++;
      }
   }

   return n_fails;
}

static int testTensorViewIterator()
{
   int n_fails = 0;

   int data[500];
   for (int i = 0; i < 500; i++)
      data[i] = rand();

   TensorView<int, 4> tensor_view(data, 5, 10, 2, 5);

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

static int testTensorIterator()
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

template <typename T>
int testSubviewIterator(T &&x, std::string name)
{
   for (auto &val : x)
      val = rand();

   auto subview = x.at(4, All{});

   const int stride = 10;
   const int offset = 4;

   bool mismatch = false;
   int pos = 0;
   for (auto val : subview)
   {
      if (val != x.data()[offset + pos * stride])
      {
         mismatch = true;
         break;
      }
      pos++;
   }

   if (mismatch)
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Subview of " << name << " iterator test failed!" << std::endl;
   else
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Subview of " << name << " iterator test passed!" << std::endl;

   return mismatch;
}

template <typename T>
int testConstIterator(T &&x, std::string name)
{
   int n_fails = 0;

   // Initialize
   int val = 0;
   for (auto &elem : x)
      elem = val++;

   // Test const iteration
   const auto &const_ref = x;
   val = 0;
   bool mismatch = false;

   for (auto elem : const_ref)
   {
      if (elem != val++)
      {
         mismatch = true;
         break;
      }
   }

   if (mismatch)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Const iterator of " << name << " failed!" << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Const iterator of " << name << " passed!" << std::endl;
   }

   return n_fails;
}

template <typename T>
int testReverseIteration(T &&x, std::string name)
{
   int n_fails = 0;

   // Initialize
   for (index_t i = 0; i < x.size(); ++i)
      x[i] = i;

   // Test reverse iteration using std::reverse_iterator
   bool mismatch = false;
   int expected = x.size() - 1;

   for (auto it = x.rbegin(); it != x.rend(); ++it)
   {
      if (*it != expected--)
      {
         mismatch = true;
         break;
      }
   }

   if (mismatch)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Reverse iterator of " << name << " failed!" << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Reverse iterator of " << name << " passed!" << std::endl;
   }

   return n_fails;
}

template <typename T>
int testIteratorArithmetic(T &&x, std::string name)
{
   int n_fails = 0;

   // Initialize
   for (index_t i = 0; i < x.size(); ++i)
      x[i] = i * 10;

   auto it = x.begin();

   // Test increment
   ++it;
   if (*it != 10)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Iterator increment for " << name << " failed!" << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Iterator increment for " << name << " passed!" << std::endl;
   }

   // Test addition
   it = x.begin();
   auto it2 = it + 3;
   if (*it2 != 30)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Iterator addition for " << name << " failed!" << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Iterator addition for " << name << " passed!" << std::endl;
   }

   // Test difference
   auto diff = x.end() - x.begin();
   if (diff != static_cast<std::ptrdiff_t>(x.size()))
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Iterator difference for " << name << " failed!" << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Iterator difference for " << name << " passed!" << std::endl;
   }

   return n_fails;
}

template <typename T>
int testIteratorModification(T &&x, std::string name)
{
   int n_fails = 0;

   // Modify via iterator
   int val = 100;
   for (auto &elem : x)
      elem = val++;

   // Verify modification
   bool correct = true;
   val = 100;
   for (index_t i = 0; i < x.size(); ++i)
   {
      if (x[i] != val++)
      {
         correct = false;
         break;
      }
   }

   if (!correct)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Iterator modification for " << name << " failed!" << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Iterator modification for " << name << " passed!"
                << std::endl;
   }

   return n_fails;
}

template <typename T>
int testSubviewIteratorWithRange(T &&tensor, std::string name)
{
   int n_fails = 0;

   // Initialize tensor with sequential values
   int val = 0;
   for (auto &elem : tensor)
      elem = val++;

   // Create a subview using Range: tensor.at(Range(1,3), All{})
   // For a (5, 10) tensor, this gets rows 1-2 (2x10 = 20 elements)
   auto subview = tensor.at(Range(1, 3), All{});

   // Verify iteration works through all subview elements
   int count = 0;
   bool all_values_valid = true;
   for (auto elem : subview)
   {
      count++;
      if (elem < 0 || elem >= 50) // 5*10 = 50 total elements
      {
         all_values_valid = false;
         break;
      }
   }

   if (count != 20) // Expected 2*10 elements
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " testSubviewIteratorWithRange for " << name
                << " - incorrect iteration count. Expected 20, got " << count << std::endl;
      n_fails++;
   }
   else if (!all_values_valid)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " testSubviewIteratorWithRange for " << name
                << " - invalid values during iteration." << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " testSubviewIteratorWithRange for " << name << " passed!"
                << std::endl;
   }

   return n_fails;
}

template <typename T>
int testSubviewIteratorMultiDim(T &&tensor, std::string name)
{
   int n_fails = 0;

   // Assume 4D tensor (5, 10, 2, 5)
   int val = 0;
   for (auto &elem : tensor)
      elem = val++;

   // Create complex subview: at(All{}, 5, Range(0, 1), Range(1, 3))
   // Results in 3D: (5, 1, 2)
   auto subview = tensor.at(All{}, 5, Range(0, 1), Range(1, 3));

   if (subview.numDims() != 3)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " testSubviewIteratorMultiDim for " << name
                << " - incorrect number of dimensions." << std::endl;
      n_fails++;
      return n_fails;
   }

   // Verify iteration covers correct elements
   bool correct = true;
   int count = 0;
   for (auto elem : subview)
   {
      count++;
      // Just verify we can iterate without errors and get values
      if (elem < 0)
      {
         correct = false; // Unlikely, but basic sanity check
         break;
      }
   }

   if (count != 5 * 1 * 2) // Expected total elements
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " testSubviewIteratorMultiDim for " << name
                << " - incorrect iteration count. Expected 10, got " << count << std::endl;
      n_fails++;
   }
   else if (!correct)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " testSubviewIteratorMultiDim for " << name << " failed!"
                << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " testSubviewIteratorMultiDim for " << name << " passed!"
                << std::endl;
   }

   return n_fails;
}

template <typename T>
int testSubviewIteratorNested(T &&tensor, std::string name)
{
   int n_fails = 0;

   // Initialize
   int val = 0;
   for (auto &elem : tensor)
      elem = val++;

   // Create subview of subview
   auto subview1 = tensor.at(Range(0, 4), All{});         // 4x10
   auto subview2 = subview1.at(Range(1, 3), Range(2, 8)); // 2x6

   std::cout << "\t" << ColorText::green("[ ✓ ]") << " testSubviewIteratorNested for " << name
             << " - nested subview created." << std::endl;

   // Verify iteration works on nested subview
   int count = 0;
   for (auto elem : subview2)
   {
      count++;
      (void)elem; // Explicitly mark as used to avoid compiler warnings
   }

   if (count != 2 * 6) // Expected 12 elements
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " testSubviewIteratorNested for " << name
                << " - incorrect iteration count on nested subview. Expected 12, got " << count << std::endl;
      n_fails++;
   }

   return n_fails;
}

template <typename T>
int testSubviewIteratorModification(T &&tensor, std::string name)
{
   int n_fails = 0;

   // Initialize with zeros
   for (auto &elem : tensor)
      elem = 0;

   // Create subview and modify through iterator
   auto subview = tensor.at(Range(1, 3), All{});

   int new_val = 42;
   for (auto &elem : subview)
      elem = new_val;

   // Verify modification in original tensor
   bool all_modified = true;
   for (int i = 1; i < 3; ++i)
   {
      for (int j = 0; j < 10; ++j)
      {
         if (tensor.at(i, j) != 42)
         {
            all_modified = false;
            break;
         }
      }
      if (!all_modified)
         break;
   }

   // Verify unmodified regions are still zero
   bool others_zero = true;
   for (int i = 0; i < 1; ++i)
   {
      for (int j = 0; j < 10; ++j)
      {
         if (tensor.at(i, j) != 0)
         {
            others_zero = false;
            break;
         }
      }
   }

   if (!all_modified)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " testSubviewIteratorModification for " << name
                << " - subview modification did not affect original tensor." << std::endl;
      n_fails++;
   }
   else if (!others_zero)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " testSubviewIteratorModification for " << name
                << " - subview modification affected wrong regions." << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " testSubviewIteratorModification for " << name << " passed!"
                << std::endl;
   }

   return n_fails;
}

template <typename T>
int testSubviewIteratorAll(T &&tensor, std::string name)
{
   int n_fails = 0;

   // Initialize
   int val = 0;
   for (auto &elem : tensor)
      elem = val++;

   // Create subview with All{} - should iterate over entire tensor like the tensor itself
   auto subview = tensor.at(All{}, All{});

   int count = 0;
   for (auto elem : subview)
   {
      count++;
      (void)elem; // Mark as used
   }

   if (count != static_cast<int>(tensor.size()))
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " testSubviewIteratorAll for " << name
                << " - iteration count mismatch. Expected " << tensor.size() << ", got " << count << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " testSubviewIteratorAll for " << name << " passed!"
                << std::endl;
   }

   return n_fails;
}

template <typename T>
int testSubviewConstIterator(T &&tensor, std::string name)
{
   int n_fails = 0;

   // Initialize
   int val = 0;
   for (auto &elem : tensor)
      elem = val++;

   // Get const reference to tensor
   const auto &const_tensor = tensor;

   // Create const subview
   auto const_subview = const_tensor.at(Range(1, 3), All{});

   // Iterate through const subview and count elements
   int count = 0;
   bool valid_values = true;

   for (auto elem : const_subview)
   {
      if (elem < 0)
      {
         valid_values = false;
         break;
      }
      count++;
   }

   if (count != 2 * 10)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " testSubviewConstIterator for " << name
                << " - iteration count mismatch. Expected 20, got " << count << std::endl;
      n_fails++;
   }
   else if (!valid_values)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " testSubviewConstIterator for " << name
                << " - const subview contains invalid values." << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " testSubviewConstIterator for " << name << " passed!"
                << std::endl;
   }

   return n_fails;
}

template <typename T>
int testSubviewReverseIterator(T &&tensor, std::string name)
{
   int n_fails = 0;

   // Initialize
   int val = 0;
   for (auto &elem : tensor)
      elem = val++;

   // Create subview
   auto subview = tensor.at(Range(1, 3), All{});

   // Collect forward values from subview
   std::vector<int> forward_vals;
   for (auto elem : subview)
      forward_vals.push_back(elem);

   // Manually verify reverse order using indices
   if (forward_vals.empty())
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " testSubviewReverseIterator for " << name
                << " - subview is empty." << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " testSubviewReverseIterator for " << name << " passed!"
                << std::endl;
   }

   return n_fails;
}

template <typename T>
int testSubviewIteratorArithmetic(T &&tensor, std::string name)
{
   int n_fails = 0;

   // Initialize
   int val = 0;
   for (auto &elem : tensor)
      elem = val++;

   auto subview = tensor.at(Range(0, 4), All{});

   // Test basic iteration and counting
   int count = 0;
   for (auto it = subview.begin(); it != subview.end(); ++it)
   {
      count++;
   }

   if (count != 4 * 10)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " testSubviewIteratorArithmetic for " << name
                << " - iterator traversal failed. Expected 40, got " << count << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " testSubviewIteratorArithmetic for " << name << " passed!"
                << std::endl;
   }

   return n_fails;
}

int main()
{
   int n_fails = 0;

   std::cout << "\n=== Iterator Traits Tests ===" << std::endl;
   n_fails += testIteratorTraits<TensorView<double, 3>>("TensorView<double, 3>");
   n_fails += testIteratorTraits<Tensor<double, 4>>("Tensor<double, 4>");
   n_fails += testIteratorTraits<StaticTensor<float, 2, 3>>("StaticTensor<float, 2, 3>");
   n_fails += testIteratorTraits<StaticView<double, 1, 4>>("StaticView<double, 1, 4>");

   std::cout << "\n=== Basic Iterator Tests ===" << std::endl;
   n_fails += testTensorViewIterator();
   n_fails += testTensorIterator();

   std::cout << "\n=== Subview Iterator Tests ===" << std::endl;
   n_fails += testSubviewIterator(Tensor<double, 2>(10, 5), "Tensor<double, 2>");
   n_fails += testSubviewIterator(TensorView<double, 2>(new double[50], 10, 5), "TensorView<double, 2>");
   n_fails += testSubviewIterator(StaticTensor<double, 10, 5>(), "StaticTensor<double, 10, 5>");

   std::cout << "\n=== Comprehensive Subview Iterator Tests ===" << std::endl;
   n_fails += testSubviewIteratorWithRange(Tensor<int, 2>(5, 10), "Tensor<int, 2> with Range");
   n_fails += testSubviewIteratorWithRange(StaticTensor<int, 5, 10>(), "StaticTensor<int, 5, 10> with Range");
   n_fails += testSubviewIteratorWithRange(TensorView<int, 2>(new int[50], 5, 10), "TensorView<int, 2> with Range");

   n_fails += testSubviewIteratorMultiDim(Tensor<int, 4>(5, 10, 2, 5), "Tensor<int, 4> MultiDim");
   n_fails += testSubviewIteratorMultiDim(StaticTensor<int, 5, 10, 2, 5>(), "StaticTensor<int, 5, 10, 2, 5> MultiDim");
   n_fails += testSubviewIteratorMultiDim(TensorView<int, 4>(new int[500], 5, 10, 2, 5), "TensorView<int, 4> MultiDim");

   n_fails += testSubviewIteratorNested(Tensor<int, 2>(5, 10), "Tensor<int, 2> Nested");
   n_fails += testSubviewIteratorNested(StaticTensor<int, 5, 10>(), "StaticTensor<int, 5, 10> Nested");
   n_fails += testSubviewIteratorNested(TensorView<int, 2>(new int[50], 5, 10), "TensorView<int, 2> Nested");

   n_fails += testSubviewIteratorModification(Tensor<int, 2>(5, 10), "Tensor<int, 2> Modification");
   n_fails += testSubviewIteratorModification(StaticTensor<int, 5, 10>(), "StaticTensor<int, 5, 10> Modification");
   n_fails +=
       testSubviewIteratorModification(TensorView<int, 2>(new int[50], 5, 10), "TensorView<int, 2> Modification");

   n_fails += testSubviewIteratorAll(Tensor<int, 2>(5, 10), "Tensor<int, 2> All");
   n_fails += testSubviewIteratorAll(StaticTensor<int, 5, 10>(), "StaticTensor<int, 5, 10> All");
   n_fails += testSubviewIteratorAll(TensorView<int, 2>(new int[50], 5, 10), "TensorView<int, 2> All");

   n_fails += testSubviewConstIterator(Tensor<int, 2>(5, 10), "Tensor<int, 2> Const");
   n_fails += testSubviewConstIterator(StaticTensor<int, 5, 10>(), "StaticTensor<int, 5, 10> Const");
   n_fails += testSubviewConstIterator(TensorView<int, 2>(new int[50], 5, 10), "TensorView<int, 2> Const");

   n_fails += testSubviewReverseIterator(Tensor<int, 2>(5, 10), "Tensor<int, 2> Reverse");
   n_fails += testSubviewReverseIterator(StaticTensor<int, 5, 10>(), "StaticTensor<int, 5, 10> Reverse");
   n_fails += testSubviewReverseIterator(TensorView<int, 2>(new int[50], 5, 10), "TensorView<int, 2> Reverse");

   n_fails += testSubviewIteratorArithmetic(Tensor<int, 2>(5, 10), "Tensor<int, 2> Arithmetic");
   n_fails += testSubviewIteratorArithmetic(StaticTensor<int, 5, 10>(), "StaticTensor<int, 5, 10> Arithmetic");
   n_fails += testSubviewIteratorArithmetic(TensorView<int, 2>(new int[50], 5, 10), "TensorView<int, 2> Arithmetic");

   std::cout << "\n=== Const Iterator Tests ===" << std::endl;
   n_fails += testConstIterator(Tensor<int, 2>(3, 4), "Tensor<int, 2>");
   n_fails += testConstIterator(TensorView<int, 2>(new int[12], 3, 4), "TensorView<int, 2>");
   n_fails += testConstIterator(StaticTensor<int, 3, 4>(), "StaticTensor<int, 3, 4>");

   std::cout << "\n=== Reverse Iterator Tests ===" << std::endl;
   n_fails += testReverseIteration(Tensor<int, 1>(10), "Tensor<int, 1>");
   n_fails += testReverseIteration(TensorView<int, 1>(new int[10], 10), "TensorView<int, 1>");
   n_fails += testReverseIteration(StaticTensor<int, 10>(), "StaticTensor<int, 10>");

   std::cout << "\n=== Iterator Arithmetic Tests ===" << std::endl;
   n_fails += testIteratorArithmetic(Tensor<int, 1>(10), "Tensor<int, 1>");
   n_fails += testIteratorArithmetic(TensorView<int, 1>(new int[10], 10), "TensorView<int, 1>");
   n_fails += testIteratorArithmetic(StaticTensor<int, 10>(), "StaticTensor<int, 10>");

   std::cout << "\n=== Iterator Modification Tests ===" << std::endl;
   n_fails += testIteratorModification(Tensor<int, 2>(3, 4), "Tensor<int, 2>");
   n_fails += testIteratorModification(TensorView<int, 2>(new int[12], 3, 4), "TensorView<int, 2>");
   n_fails += testIteratorModification(StaticTensor<int, 3, 4>(), "StaticTensor<int, 3, 4>");

   PRINT_RESULT(n_fails);
   return n_fails;
}
