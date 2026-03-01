#include "test.hpp"
using namespace tensor;

template <typename T>
int testSubviewMixedIndexing(std::string name, T &&x)
{
   int n_fails = 0;

   for (auto &val : x)
      val = rand();

   // Test: All(), single index, Range, Range
   auto subview = x.at(All(), 2, Range(0, 1), Range(2, 4));

   if (subview.numDims() != 3)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " mixed indexing for " << name
                << " produced view with wrong number of dimensions."
                << " Expected 3, got " << subview.numDims() << "." << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " mixed indexing for " << name
                << " produced view with correct number of dimensions." << std::endl;
   }

   bool mismatch_found = false;
   for (int i = 0; i < 5; ++i)
   {
      for (int j = 0; j < 1; ++j)
      {
         for (int k = 0; k < 2; ++k)
         {
            if (subview.at(i, j, k) != x.at(i, 2, j, 2 + k))
            {
               mismatch_found = true;
               break;
            }
         }
      }
   }

   if (mismatch_found)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " mixed indexing produced incorrect view." << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " mixed indexing produced correct view." << std::endl;
   }

   return n_fails;
}

template <typename T>
int testSubviewMultipleRanges(std::string name, T &&x)
{
   int n_fails = 0;

   for (auto &val : x)
      val = rand();

   // Test: Multiple Range operations
   auto subview = x.at(Range(1, 4), Range(0, 5), 0, Range(1, 4));

   if (subview.numDims() != 3)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " multiple Range indexing for " << name
                << " produced view with wrong dimensions." << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " multiple Range indexing for " << name
                << " produced correct dimensions." << std::endl;
   }

   bool mismatch_found = false;
   for (int i = 0; i < 3; ++i)
   {
      for (int j = 0; j < 5; ++j)
      {
         for (int k = 0; k < 3; ++k)
         {
            if (subview.at(i, j, k) != x.at(i + 1, j, 0, k + 1))
            {
               mismatch_found = true;
               break;
            }
         }
      }
   }

   if (mismatch_found)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " multiple Range indexing produced incorrect view." << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " multiple Range indexing produced correct view." << std::endl;
   }

   return n_fails;
}

template <typename T>
int testSubviewSingleRange(std::string name, T &&x)
{
   int n_fails = 0;

   for (auto &val : x)
      val = rand();

   // Test: Range on first dimension only
   auto subview = x.at(Range(1, 3), All(), All(), All());

   if (subview.numDims() != 4)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " single Range indexing for " << name
                << " produced wrong dimensions." << std::endl;
      n_fails++;
   }

   bool mismatch_found = false;
   for (int i = 0; i < 2; ++i)
   {
      for (int j = 0; j < 10; ++j)
      {
         for (int k = 0; k < 2; ++k)
         {
            for (int l = 0; l < 5; ++l)
            {
               if (subview.at(i, j, k, l) != x.at(i + 1, j, k, l))
               {
                  mismatch_found = true;
                  break;
               }
            }
         }
      }
   }

   if (mismatch_found)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " single Range indexing produced incorrect view." << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " single Range indexing produced correct view." << std::endl;
   }

   return n_fails;
}

template <typename T>
int testSubviewModifications(std::string name, T &&x)
{
   int n_fails = 0;

   for (auto &val : x)
      val = 0;

   // Test: Modify subview and verify original is modified
   auto subview = x.at(Range(1, 3), 5, All(), All());

   if (subview.numDims() != 3)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " modification test for " << name << " produced wrong dimensions."
                << std::endl;
      n_fails++;
   }

   // Modify subview
   for (int i = 0; i < 2; ++i)
   {
      for (int j = 0; j < 2; ++j)
      {
         for (int k = 0; k < 5; ++k)
         {
            subview.at(i, j, k) = 42;
         }
      }
   }

   // Check if original was modified
   bool all_modified = true;
   for (int i = 1; i < 3; ++i)
   {
      for (int j = 0; j < 2; ++j)
      {
         for (int k = 0; k < 5; ++k)
         {
            if (x.at(i, 5, j, k) != 42)
            {
               all_modified = false;
               break;
            }
         }
      }
   }

   if (!all_modified)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " subview modifications for " << name
                << " did not affect original tensor." << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " subview modifications for " << name
                << " correctly modified original tensor." << std::endl;
   }

   return n_fails;
}

template <typename T>
int testSubviewDimensions(std::string name, T &&x)
{
   int n_fails = 0;

   for (auto &val : x)
      val = rand();

   // Test: Check shape of subview
   auto subview = x.at(Range(0, 4), Range(2, 8), 1, Range(0, 3));

   // Expected shape: (4, 6, 3) - reduced from (5, 10, 2, 5)
   if (subview.shape(0) != 4 || subview.shape(1) != 6 || subview.shape(2) != 3)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " dimension check for " << name << " failed. Shape mismatch."
                << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " dimension check for " << name << " passed. Correct shape."
                << std::endl;
   }

   return n_fails;
}

template <typename T>
int testSubviewEdgeCases(std::string name, T &&x)
{
   int n_fails = 0;

   for (auto &val : x)
      val = rand();

   // Test: Range at boundaries
   auto subview1 = x.at(0, All(), All(), All());
   if (subview1.numDims() != 3)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " edge case (first index) for " << name << " failed."
                << std::endl;
      n_fails++;
   }

   auto subview2 = x.at(4, All(), All(), All());
   if (subview2.numDims() != 3)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " edge case (last index) for " << name << " failed." << std::endl;
      n_fails++;
   }

   // Test: Range with minimal extent
   auto subview3 = x.at(Range(2, 3), All(), All(), All());
   if (subview3.numDims() != 4 || subview3.shape(0) != 1)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " edge case (Range with 1-element) for " << name << " failed."
                << std::endl;
      n_fails++;
   }

   if (n_fails == 0)
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " edge cases for " << name << " passed." << std::endl;
   }

   return n_fails;
}

template <typename T>
int testSubviewAssignment1d(std::string name, T &&x)
{
   int n_fails = 0;

   // Initialize tensor with zeros
   for (auto &val : x)
      val = 0;

   // Create a source tensor for assignment (shape matches 4th dimension)
   Tensor<int, 1> source(5);
   for (int i = 0; i < 5; ++i)
      source[i] = 100 + i;

   // Assign to a 1D subview (fixing first 3 dimensions)
   x.at(0, 0, 0, All()) = source;

   // Verify assignment
   bool correct = true;
   for (int i = 0; i < 5; ++i)
   {
      if (x.at(0, 0, 0, i) != 100 + i)
      {
         correct = false;
         break;
      }
   }

   // Verify other elements are still zero
   if (x.at(1, 0, 0, 0) != 0 || x.at(0, 1, 0, 0) != 0)
      correct = false;

   if (!correct)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " 1D subview assignment for " << name << " failed!" << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " 1D subview assignment for " << name << " passed!"
                << std::endl;
   }

   return n_fails;
}

template <typename T>
int testSubviewAssignment2d(std::string name, T &&x)
{
   int n_fails = 0;

   // Initialize with zeros
   for (auto &val : x)
      val = 0;

   // Create 2D source (10x5 - matches 2nd and 4th dimensions)
   Tensor<int, 2> source(10, 5);
   for (int i = 0; i < 10; ++i)
      for (int j = 0; j < 5; ++j)
         source(i, j) = 200 + i * 10 + j;

   // Assign to 2D subview
   x.at(0, All(), 0, All()) = source;

   // Verify assignment
   bool correct = true;
   for (int i = 0; i < 10; ++i)
   {
      for (int j = 0; j < 5; ++j)
      {
         if (x.at(0, i, 0, j) != 200 + i * 10 + j)
         {
            correct = false;
            break;
         }
      }
   }

   // Verify other elements are still zero
   if (x.at(1, 0, 0, 0) != 0 || x.at(0, 0, 1, 0) != 0)
      correct = false;

   if (!correct)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " 2D subview assignment for " << name << " failed!" << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " 2D subview assignment for " << name << " passed!"
                << std::endl;
   }

   return n_fails;
}

template <typename T>
int testSubviewAssignmentRange(std::string name, T &&x)
{
   int n_fails = 0;

   // Initialize with zeros
   for (auto &val : x)
      val = 0;

   // Create source (3x10x5) to match Range(1,4) x All() x All()
   Tensor<int, 3> source(3, 10, 5);
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 10; ++j)
         for (int k = 0; k < 5; ++k)
            source(i, j, k) = 300 + i * 100 + j * 10 + k;

   // Assign to Range-based subview
   x.at(Range(1, 4), All(), 0, All()) = source;

   // Verify assignment
   bool correct = true;
   for (int i = 0; i < 3; ++i)
   {
      for (int j = 0; j < 10; ++j)
      {
         for (int k = 0; k < 5; ++k)
         {
            int expected = 300 + i * 100 + j * 10 + k;
            if (x.at(1 + i, j, 0, k) != expected)
            {
               correct = false;
               break;
            }
         }
      }
   }

   // Verify elements outside range are still zero
   if (x.at(0, 0, 0, 0) != 0)
      correct = false;

   if (!correct)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Range subview assignment for " << name << " failed!"
                << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Range subview assignment for " << name << " passed!"
                << std::endl;
   }

   return n_fails;
}

template <typename T>
int testSubviewToSubviewAssignment(std::string name, T &&x)
{
   int n_fails = 0;

   // Initialize with known pattern
   for (int i = 0; i < 5; ++i)
      for (int j = 0; j < 10; ++j)
         for (int k = 0; k < 2; ++k)
            for (int l = 0; l < 5; ++l)
               x.at(i, j, k, l) = i * 1000 + j * 100 + k * 10 + l;

   // Copy one subview to another subview
   // Copy x.at(0, Range(0,5), 0, All()) to x.at(1, Range(5,10), 0, All())
   auto source_subview = x.at(0, Range(0, 5), 0, All());
   auto target_subview = x.at(1, Range(5, 10), 0, All());

   target_subview = source_subview;

   // Verify the copy
   bool correct = true;
   for (int i = 0; i < 5; ++i)
   {
      for (int j = 0; j < 5; ++j)
      {
         int expected = 0 * 1000 + i * 100 + 0 * 10 + j;
         if (x.at(1, 5 + i, 0, j) != expected)
         {
            correct = false;
            break;
         }
      }
   }

   if (!correct)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Subview-to-subview assignment for " << name << " failed!"
                << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Subview-to-subview assignment for " << name << " passed!"
                << std::endl;
   }

   return n_fails;
}

template <typename T>
int testSubviewComplexAssignment(std::string name, T &&x)
{
   int n_fails = 0;

   // Initialize with zeros
   for (auto &val : x)
      val = 0;

   // Create a 3D source (3x4x3)
   Tensor<int, 3> source(3, 4, 3);
   for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 4; ++j)
         for (int k = 0; k < 3; ++k)
            source(i, j, k) = 500 + i * 100 + j * 10 + k;

   // Assign to complex 3D subview with Range operations
   x.at(Range(1, 4), Range(2, 6), 1, Range(1, 4)) = source;

   // Verify assignment
   bool correct = true;
   for (int i = 0; i < 3; ++i)
   {
      for (int j = 0; j < 4; ++j)
      {
         for (int k = 0; k < 3; ++k)
         {
            if (x.at(1 + i, 2 + j, 1, 1 + k) != source(i, j, k))
            {
               correct = false;
               break;
            }
         }
      }
   }

   // Verify elements outside the subview are still zero
   if (x.at(0, 0, 0, 0) != 0 || x.at(0, 2, 1, 1) != 0 || x.at(1, 1, 1, 1) != 0)
      correct = false;

   if (!correct)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Complex subview assignment for " << name << " failed!"
                << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Complex subview assignment for " << name << " passed!"
                << std::endl;
   }

   return n_fails;
}

template <typename T>
int testSubviewOverlappingAssignment(std::string name, T &&x)
{
   int n_fails = 0;

   // Initialize with sequence
   int val = 0;
   for (auto &elem : x)
      elem = val++;

   // Create source from a subview of the same tensor
   auto source = x.at(Range(0, 2), Range(0, 5), 1, All());

   // Store expected values
   std::vector<int> expected_values;
   for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 5; ++j)
         for (int k = 0; k < 5; ++k)
            expected_values.push_back(x.at(i, j, 1, k));

   // Assign to different location in same tensor
   x.at(Range(3, 5), Range(5, 10), 0, All()) = source;

   // Verify assignment
   bool correct = true;
   int idx = 0;
   for (int i = 0; i < 2; ++i)
   {
      for (int j = 0; j < 5; ++j)
      {
         for (int k = 0; k < 5; ++k)
         {
            if (x.at(3 + i, 5 + j, 0, k) != expected_values[idx++])
            {
               correct = false;
               break;
            }
         }
      }
   }

   if (!correct)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Overlapping subview assignment for " << name << " failed!"
                << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Overlapping subview assignment for " << name << " passed!"
                << std::endl;
   }

   return n_fails;
}

template <typename T>
int testComplexSubviewOfComplexSubview(std::string name, T &&x)
{
   int n_fails = 0;

   // Initialize with sequential values for easy verification
   int val = 0;
   for (auto &elem : x)
      elem = val++;

   // Create first-level complex subview with stride: All(), Range(0, 10, 2), All(), Range(1, 5)
   // Tensor shape: (5, 10, 2, 5)
   // Result: (5, 5, 2, 4) because Range(0,10,2) gives [0,2,4,6,8]=5, Range(1,5) gives [1,2,3,4]=4
   auto subview1 = x.at(All(), Range(0, 10, 2), All(), Range(1, 5));

   index_t expected_size1 = 5 * 5 * 2 * 4;
   if (subview1.numDims() != 4 || subview1.size() != expected_size1)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Complex subview of complex subview for " << name
                << " - first level subview has wrong size! Expected " << expected_size1 << ", got " << subview1.size()
                << std::endl;
      n_fails++;
      return n_fails;
   }

   // Create second-level complex subview from first: Range(1, 5), 2, All(), Range(0, 3)
   // subview1 is (5, 5, 2, 4)
   // Result: (4, 2, 3) removing dim 1, keeping dims 0,2,3
   auto subview2 = subview1.at(Range(1, 5), 2, All(), Range(0, 3));

   index_t expected_size2 = 4 * 2 * 3;
   if (subview2.numDims() != 3 || subview2.size() != expected_size2)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Complex subview of complex subview for " << name
                << " - second level subview has wrong size! Expected " << expected_size2 << ", got " << subview2.size()
                << std::endl;
      n_fails++;
      return n_fails;
   }

   // Verify subview2 data correctness by checking against original tensor
   // subview2.at(i, j, k) should map to x.at(1+i, 0+2*2, j, 1+k) = x.at(1+i, 4, j, 1+k)
   bool data_correct = true;
   for (int i = 0; i < 4 && data_correct; ++i)
   {
      for (int j = 0; j < 2 && data_correct; ++j)
      {
         for (int k = 0; k < 3 && data_correct; ++k)
         {
            if (subview2.at(i, j, k) != x.at(1 + i, 4, j, 1 + k))
               data_correct = false;
         }
      }
   }

   if (!data_correct)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Complex subview of complex subview for " << name
                << " - data mismatch in nested subview!" << std::endl;
      n_fails++;
      return n_fails;
   }

   // Verify we can iterate through the nested subview
   index_t count = 0;
   for (auto elem : subview2)
   {
      (void)elem;
      count++;
   }

   if (count != subview2.size())
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Complex subview of complex subview for " << name
                << " - iteration count mismatch! Expected " << subview2.size() << ", got " << count << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Complex subview of complex subview for " << name
                << " - nested subview with strides works correctly!" << std::endl;
   }

   return n_fails;
}

template <typename T>
int testNestedSubviewModification(std::string name, T &&x)
{
   int n_fails = 0;

   // Initialize all to zero
   for (auto &elem : x)
      elem = 0;

   // Create first-level subview with stride: Range(0, 4), Range(1, 10, 2), All(), Range(0, 5)
   // Stride 2 on dimension 1 gives non-contiguous access
   auto subview1 = x.at(Range(0, 4), Range(1, 10, 2), All(), Range(0, 5));

   // Create second-level subview: 1 (fixed index), Range(0, 4, 2), 0 (fixed), All()
   // Results in a 1D view
   auto subview2 = subview1.at(1, Range(0, 4, 2), 0, All());

   // Modify through nested subview
   int new_val = 42;
   for (auto &elem : subview2)
      elem = new_val;

   // Verify at least some modifications occurred by checking original tensor
   bool at_least_one_modified = false;
   for (auto elem : x)
   {
      if (elem == 42)
      {
         at_least_one_modified = true;
         break;
      }
   }

   // Check some regions that shouldn't be modified (dimensions we didn't index)
   bool unmodified_preserved = (x.at(0, 0, 0, 0) == 0) && (x.at(0, 0, 1, 1) == 0);

   if (!at_least_one_modified)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Nested subview modification for " << name
                << " - no modifications detected!" << std::endl;
      n_fails++;
   }
   else if (!unmodified_preserved)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Nested subview modification for " << name
                << " - unrelated regions were modified!" << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Nested subview modification for " << name
                << " - modifications with strides propagated correctly!" << std::endl;
   }

   return n_fails;
}

template <typename T>
int testTripleNestedSubview(std::string name, T &&x)
{
   int n_fails = 0;

   // Initialize with sequence
   int val = 0;
   for (auto &elem : x)
      elem = val++;

   // Level 1: Create simple complex subview: Range(0, 4), All(), 1, Range(0, 4)
   // x is (5, 10, 2, 5), result is 3D: (4, 10, 4)
   auto level1 = x.at(Range(0, 4), All(), 1, Range(0, 4));

   if (level1.numDims() != 3)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Triple nested subview for " << name
                << " - level 1 has wrong dimensions! Got " << level1.numDims() << std::endl;
      n_fails++;
      return n_fails;
   }

   // Level 2: Create subview: Range(1, 3), Range(2, 7), 0
   // level1 is 3D (4, 10, 4), result is 2D: (2, 5)
   auto level2 = level1.at(Range(1, 3), Range(2, 7), 0);

   if (level2.numDims() != 2)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Triple nested subview for " << name
                << " - level 2 has wrong dimensions! Got " << level2.numDims() << std::endl;
      n_fails++;
      return n_fails;
   }

   // Level 3: Create final subview: 1, Range(0, 3)
   // level2 is 2D (2, 5), result is 1D: (3)
   auto level3 = level2.at(1, Range(0, 3));

   if (level3.numDims() != 1)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Triple nested subview for " << name
                << " - level 3 has wrong dimensions! Got " << level3.numDims() << std::endl;
      n_fails++;
      return n_fails;
   }

   // Verify we can iterate and get correct size
   index_t count = 0;
   for (auto elem : level3)
   {
      (void)elem;
      count++;
   }

   if (count != level3.size())
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Triple nested subview for " << name
                << " - iteration count mismatch! Expected " << level3.size() << ", got " << count << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Triple nested subview for " << name
                << " - triple nesting works correctly!" << std::endl;
   }

   return n_fails;
}

template <typename T>
int testNestedSubviewAssignment(std::string name, T &&x)
{
   int n_fails = 0;

   // Initialize with zeros
   for (auto &elem : x)
      elem = 0;

   // Get first-level subview: Range(0, 5), Range(1, 10, 2), All(), Range(0, 5)
   // This is 4D: (5, 5, 2, 5)
   auto subview1 = x.at(Range(0, 5), Range(1, 10, 2), All(), Range(0, 5));

   // Get second-level subview: 2 (fixed), Range(0, 4, 2), 0 (fixed), All()
   // Result is 2D after removing 2 dimensions: (2, 5)
   auto subview2 = subview1.at(2, Range(0, 4, 2), 0, All());

   if (subview2.numDims() != 2)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Nested subview assignment for " << name
                << " - second level has wrong dimensions! Got " << subview2.numDims() << std::endl;
      n_fails++;
      return n_fails;
   }

   // Try to modify through nested subview
   for (auto &elem : subview2)
      elem = 77;

   // Verify modifications reached the original tensor
   bool at_least_one_modified = false;
   for (auto elem : x)
   {
      if (elem == 77)
      {
         at_least_one_modified = true;
         break;
      }
   }

   if (!at_least_one_modified)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Nested subview assignment for " << name
                << " - modifications did not propagate!" << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Nested subview assignment for " << name
                << " with strides passed!" << std::endl;
   }

   return n_fails;
}

template <typename T>
int testComplexSubviewCorrectness(std::string name, T &&x)
{
   int n_fails = 0;

   // Initialize with a known pattern for verification
   int val = 1;
   for (auto &elem : x)
      elem = val++;

   // Test case 1: Create a complex subview with stride and fixed index
   // x is (5, 10, 2, 5)
   // x.at(All(), Range(0, 10, 2), 1, All())
   // Result: (5, 5, 5) because Range(0,10,2) gives [0,2,4,6,8]=5 elements
   auto sv1 = x.at(All(), Range(0, 10, 2), 1, All());

   if (sv1.numDims() != 3)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Complex subview correctness test 1 for " << name
                << " - sv1 has wrong dims! Expected 3, got " << sv1.numDims() << std::endl;
      n_fails++;
      return n_fails;
   }

   index_t expected_size1 = 5 * 5 * 5; // 3D result = 125
   if (sv1.size() != expected_size1)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Complex subview correctness test 1 for " << name
                << " - subview has wrong size! Expected " << expected_size1 << ", got " << sv1.size() << std::endl;
      n_fails++;
      return n_fails;
   }

   // Verify sv1 data correctness: sv1 should map to x.at(i, 0+2*j, 1, k) for all i,j,k
   bool sv1_correct = true;
   for (int i = 0; i < 5 && sv1_correct; ++i)
   {
      for (int j = 0; j < 5 && sv1_correct; ++j)
      {
         for (int k = 0; k < 5 && sv1_correct; ++k)
         {
            if (sv1.at(i, j, k) != x.at(i, 0 + 2 * j, 1, k))
               sv1_correct = false;
         }
      }
   }

   if (!sv1_correct)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Complex subview correctness test 1 for " << name
                << " - sv1 data mismatch with original tensor!" << std::endl;
      n_fails++;
      return n_fails;
   }

   // Test case 2: Create a subview from sv1 WITH strides to verify stride-on-stride composition
   // sv1 is (5, 5, 5)
   // sv1.at(Range(0, 5, 2), Range(1, 5, 2), All())
   // Result: (3, 2, 5) because Range(0,5,2) gives [0,2,4]=3, Range(1,5,2) gives [1,3]=2
   auto sv2 = sv1.at(Range(0, 5, 2), Range(1, 5, 2), All());

   if (sv2.numDims() != 3)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Complex subview correctness test 2 for " << name
                << " - sv2 has wrong dims! Expected 3, got " << sv2.numDims() << std::endl;
      n_fails++;
      return n_fails;
   }

   index_t expected_size2 = 3 * 2 * 5; // 3D result = 30
   if (sv2.size() != expected_size2)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Complex subview correctness test 2 for " << name
                << " - nested subview has wrong size! Expected " << expected_size2 << ", got " << sv2.size()
                << std::endl;
      n_fails++;
      return n_fails;
   }

   // Verify sv2 data correctness with stride-on-stride composition:
   // sv2.at(i, j, k) should map to:
   // sv1.at(0+2*i, 1+2*j, k) which maps to x.at(0+2*i, 0+2*(1+2*j), 1, k) = x.at(2*i, 2+4*j, 1, k)
   bool sv2_correct = true;
   for (int i = 0; i < 3 && sv2_correct; ++i)
   {
      for (int j = 0; j < 2 && sv2_correct; ++j)
      {
         for (int k = 0; k < 5 && sv2_correct; ++k)
         {
            if (sv2.at(i, j, k) != x.at(2 * i, 2 + 4 * j, 1, k))
               sv2_correct = false;
         }
      }
   }

   if (!sv2_correct)
   {
      std::cout << "\t" << ColorText::red("[ ✗ ]") << " Complex subview correctness test 2 for " << name
                << " - sv2 data mismatch with stride-on-stride composition!" << std::endl;
      n_fails++;
   }
   else
   {
      std::cout << "\t" << ColorText::green("[ ✓ ]") << " Complex subview correctness for " << name
                << " with nested strides works correctly!" << std::endl;
   }

   std::cout.flush();
   return n_fails;
}

int main()
{
   int n_fails = 0;

   // Test container types
   std::vector<std::string> tensor_types = {"Tensor<int, 4>", "TensorView<int, 4>", "StaticTensor<int, 5, 10, 2, 5>",
                                            "CStaticView<int, 5, 10, 2, 5>"};

   std::cout << "\n=== Mixed Indexing Tests (All, Index, Range, Range) ===" << std::endl;
   n_fails += testSubviewMixedIndexing("Tensor<int, 4>", makeTensor<int>(5, 10, 2, 5));
   n_fails += testSubviewMixedIndexing("TensorView<int, 4>", TensorView<int, 4>(new int[500], 5, 10, 2, 5));
   n_fails += testSubviewMixedIndexing("StaticTensor<int, 5, 10, 2, 5>", StaticTensor<int, 5, 10, 2, 5>());
   n_fails += testSubviewMixedIndexing("CStaticView<int, 5, 10, 2, 5>", CStaticView<int, 5, 10, 2, 5>(new int[500]));

   std::cout << "\n=== Multiple Ranges Tests ===" << std::endl;
   n_fails += testSubviewMultipleRanges("Tensor<int, 4>", makeTensor<int>(5, 10, 2, 5));
   n_fails += testSubviewMultipleRanges("TensorView<int, 4>", TensorView<int, 4>(new int[500], 5, 10, 2, 5));
   n_fails += testSubviewMultipleRanges("StaticTensor<int, 5, 10, 2, 5>", StaticTensor<int, 5, 10, 2, 5>());
   n_fails += testSubviewMultipleRanges("CStaticView<int, 5, 10, 2, 5>", CStaticView<int, 5, 10, 2, 5>(new int[500]));

   std::cout << "\n=== Single Range Tests (Range on first dimension) ===" << std::endl;
   n_fails += testSubviewSingleRange("Tensor<int, 4>", makeTensor<int>(5, 10, 2, 5));
   n_fails += testSubviewSingleRange("TensorView<int, 4>", TensorView<int, 4>(new int[500], 5, 10, 2, 5));
   n_fails += testSubviewSingleRange("StaticTensor<int, 5, 10, 2, 5>", StaticTensor<int, 5, 10, 2, 5>());
   n_fails += testSubviewSingleRange("CStaticView<int, 5, 10, 2, 5>", CStaticView<int, 5, 10, 2, 5>(new int[500]));

   std::cout << "\n=== Subview Modification Tests ===" << std::endl;
   n_fails += testSubviewModifications("Tensor<int, 4>", makeTensor<int>(5, 10, 2, 5));
   n_fails += testSubviewModifications("TensorView<int, 4>", TensorView<int, 4>(new int[500], 5, 10, 2, 5));
   n_fails += testSubviewModifications("StaticTensor<int, 5, 10, 2, 5>", StaticTensor<int, 5, 10, 2, 5>());
   n_fails += testSubviewModifications("CStaticView<int, 5, 10, 2, 5>", CStaticView<int, 5, 10, 2, 5>(new int[500]));

   std::cout << "\n=== Dimension/Shape Tests ===" << std::endl;
   n_fails += testSubviewDimensions("Tensor<int, 4>", makeTensor<int>(5, 10, 2, 5));
   n_fails += testSubviewDimensions("TensorView<int, 4>", TensorView<int, 4>(new int[500], 5, 10, 2, 5));
   n_fails += testSubviewDimensions("StaticTensor<int, 5, 10, 2, 5>", StaticTensor<int, 5, 10, 2, 5>());
   n_fails += testSubviewDimensions("CStaticView<int, 5, 10, 2, 5>", CStaticView<int, 5, 10, 2, 5>(new int[500]));

   std::cout << "\n=== Edge Cases Tests ===" << std::endl;
   n_fails += testSubviewEdgeCases("Tensor<int, 4>", makeTensor<int>(5, 10, 2, 5));
   n_fails += testSubviewEdgeCases("TensorView<int, 4>", TensorView<int, 4>(new int[500], 5, 10, 2, 5));
   n_fails += testSubviewEdgeCases("StaticTensor<int, 5, 10, 2, 5>", StaticTensor<int, 5, 10, 2, 5>());
   n_fails += testSubviewEdgeCases("CStaticView<int, 5, 10, 2, 5>", CStaticView<int, 5, 10, 2, 5>(new int[500]));

   std::cout << "\n=== Subview Assignment Tests (1D) ===" << std::endl;
   n_fails += testSubviewAssignment1d("Tensor<int, 4>", makeTensor<int>(5, 10, 2, 5));
   n_fails += testSubviewAssignment1d("TensorView<int, 4>", TensorView<int, 4>(new int[500], 5, 10, 2, 5));
   n_fails += testSubviewAssignment1d("StaticTensor<int, 5, 10, 2, 5>", StaticTensor<int, 5, 10, 2, 5>());
   n_fails += testSubviewAssignment1d("CStaticView<int, 5, 10, 2, 5>", CStaticView<int, 5, 10, 2, 5>(new int[500]));

   std::cout << "\n=== Subview Assignment Tests (2D) ===" << std::endl;
   n_fails += testSubviewAssignment2d("Tensor<int, 4>", makeTensor<int>(5, 10, 2, 5));
   n_fails += testSubviewAssignment2d("TensorView<int, 4>", TensorView<int, 4>(new int[500], 5, 10, 2, 5));
   n_fails += testSubviewAssignment2d("StaticTensor<int, 5, 10, 2, 5>", StaticTensor<int, 5, 10, 2, 5>());
   n_fails += testSubviewAssignment2d("CStaticView<int, 5, 10, 2, 5>", CStaticView<int, 5, 10, 2, 5>(new int[500]));

   std::cout << "\n=== Subview Assignment Tests (Range) ===" << std::endl;
   n_fails += testSubviewAssignmentRange("Tensor<int, 4>", makeTensor<int>(5, 10, 2, 5));
   n_fails += testSubviewAssignmentRange("TensorView<int, 4>", TensorView<int, 4>(new int[500], 5, 10, 2, 5));
   n_fails += testSubviewAssignmentRange("StaticTensor<int, 5, 10, 2, 5>", StaticTensor<int, 5, 10, 2, 5>());
   n_fails += testSubviewAssignmentRange("CStaticView<int, 5, 10, 2, 5>", CStaticView<int, 5, 10, 2, 5>(new int[500]));

   std::cout << "\n=== Subview-to-Subview Assignment Tests ===" << std::endl;
   n_fails += testSubviewToSubviewAssignment("Tensor<int, 4>", makeTensor<int>(5, 10, 2, 5));
   n_fails += testSubviewToSubviewAssignment("TensorView<int, 4>", TensorView<int, 4>(new int[500], 5, 10, 2, 5));
   n_fails += testSubviewToSubviewAssignment("StaticTensor<int, 5, 10, 2, 5>", StaticTensor<int, 5, 10, 2, 5>());
   n_fails +=
       testSubviewToSubviewAssignment("CStaticView<int, 5, 10, 2, 5>", CStaticView<int, 5, 10, 2, 5>(new int[500]));

   std::cout << "\n=== Complex Subview Assignment Tests ===" << std::endl;
   n_fails += testSubviewComplexAssignment("Tensor<int, 4>", makeTensor<int>(5, 10, 2, 5));
   n_fails += testSubviewComplexAssignment("TensorView<int, 4>", TensorView<int, 4>(new int[500], 5, 10, 2, 5));
   n_fails += testSubviewComplexAssignment("StaticTensor<int, 5, 10, 2, 5>", StaticTensor<int, 5, 10, 2, 5>());
   n_fails +=
       testSubviewComplexAssignment("CStaticView<int, 5, 10, 2, 5>", CStaticView<int, 5, 10, 2, 5>(new int[500]));

   std::cout << "\n=== Overlapping Subview Assignment Tests ===" << std::endl;
   n_fails += testSubviewOverlappingAssignment("Tensor<int, 4>", makeTensor<int>(5, 10, 2, 5));
   n_fails += testSubviewOverlappingAssignment("TensorView<int, 4>", TensorView<int, 4>(new int[500], 5, 10, 2, 5));
   n_fails += testSubviewOverlappingAssignment("StaticTensor<int, 5, 10, 2, 5>", StaticTensor<int, 5, 10, 2, 5>());
   n_fails +=
       testSubviewOverlappingAssignment("CStaticView<int, 5, 10, 2, 5>", CStaticView<int, 5, 10, 2, 5>(new int[500]));

   std::cout << "\n=== Complex Subview of Complex Subview Tests ===" << std::endl;
   n_fails += testComplexSubviewOfComplexSubview("Tensor<int, 4>", makeTensor<int>(5, 10, 2, 5));
   n_fails += testComplexSubviewOfComplexSubview("TensorView<int, 4>", TensorView<int, 4>(new int[500], 5, 10, 2, 5));
   n_fails += testComplexSubviewOfComplexSubview("StaticTensor<int, 5, 10, 2, 5>", StaticTensor<int, 5, 10, 2, 5>());
   n_fails +=
       testComplexSubviewOfComplexSubview("CStaticView<int, 5, 10, 2, 5>", CStaticView<int, 5, 10, 2, 5>(new int[500]));

   std::cout << "\n=== Nested Subview Modification Tests ===" << std::endl;
   n_fails += testNestedSubviewModification("Tensor<int, 4>", makeTensor<int>(5, 10, 2, 5));
   n_fails += testNestedSubviewModification("TensorView<int, 4>", TensorView<int, 4>(new int[500], 5, 10, 2, 5));
   n_fails += testNestedSubviewModification("StaticTensor<int, 5, 10, 2, 5>", StaticTensor<int, 5, 10, 2, 5>());
   n_fails +=
       testNestedSubviewModification("CStaticView<int, 5, 10, 2, 5>", CStaticView<int, 5, 10, 2, 5>(new int[500]));

   std::cout << "\n=== Triple Nested Subview Tests ===" << std::endl;
   n_fails += testTripleNestedSubview("Tensor<int, 4>", makeTensor<int>(5, 10, 2, 5));
   n_fails += testTripleNestedSubview("TensorView<int, 4>", TensorView<int, 4>(new int[500], 5, 10, 2, 5));
   n_fails += testTripleNestedSubview("StaticTensor<int, 5, 10, 2, 5>", StaticTensor<int, 5, 10, 2, 5>());
   n_fails += testTripleNestedSubview("CStaticView<int, 5, 10, 2, 5>", CStaticView<int, 5, 10, 2, 5>(new int[500]));

   std::cout << "\n=== Nested Subview Assignment Tests ===" << std::endl;
   n_fails += testNestedSubviewAssignment("Tensor<int, 4>", makeTensor<int>(5, 10, 2, 5));
   n_fails += testNestedSubviewAssignment("TensorView<int, 4>", TensorView<int, 4>(new int[500], 5, 10, 2, 5));
   n_fails += testNestedSubviewAssignment("StaticTensor<int, 5, 10, 2, 5>", StaticTensor<int, 5, 10, 2, 5>());
   n_fails += testNestedSubviewAssignment("CStaticView<int, 5, 10, 2, 5>", CStaticView<int, 5, 10, 2, 5>(new int[500]));

   std::cout << "\n=== Complex Subview Correctness Tests ===" << std::endl;
   n_fails += testComplexSubviewCorrectness("Tensor<int, 4>", makeTensor<int>(5, 10, 2, 5));
   n_fails += testComplexSubviewCorrectness("TensorView<int, 4>", TensorView<int, 4>(new int[500], 5, 10, 2, 5));
   n_fails += testComplexSubviewCorrectness("StaticTensor<int, 5, 10, 2, 5>", StaticTensor<int, 5, 10, 2, 5>());
   n_fails +=
       testComplexSubviewCorrectness("CStaticView<int, 5, 10, 2, 5>", CStaticView<int, 5, 10, 2, 5>(new int[500]));

   PRINT_RESULT(n_fails);
   return n_fails;
}
