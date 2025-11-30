#pragma once
#include "TensorView/Macros.hpp"

namespace tensor::details
{
   /**
    * @brief A tensor-like structure used to initialize tensors from nested initializer lists.
    */
   template <typename T, index_t NumDims>
   struct InitializerTensor;

   template <typename T>
   struct InitializerTensor<T, 1>
   {
      using ListType = std::initializer_list<T>;
      const ListType data;

      constexpr InitializerTensor() = default;
      constexpr InitializerTensor(ListType list) : data(list) {}

      static constexpr index_t numDims()
      {
         return 1;
      }

      constexpr index_t size() const
      {
         return data.size();
      }

      constexpr index_t shape([[maybe_unused]] index_t dim) const
      {
         TENSOR_DEBUG_ASSERT(
             dim == 0, printf("Dimension %ju out of bounds for 1D initializer tensor.\n", static_cast<uintmax_t>(dim)));

         return data.size();
      }

      constexpr const T &operator[](index_t i) const
      {
         return *(std::next(data.begin(), i));
      }
   };

   template <typename T, index_t NumDims>
   struct InitializerTensor
   {
      static_assert(NumDims > 1, "NumDims must be greater than 1 for this specialization.");

      using SubType = InitializerTensor<T, NumDims - 1>;
      using ListType = std::initializer_list<SubType>;
      const ListType data;

      constexpr InitializerTensor() = default;

      constexpr InitializerTensor(ListType list) : data(list)
      {
         const index_t first_shape = data.begin()->shape(0);
         bool shapes_consistent = true;
         for (const auto &sublist : data)
         {
            shapes_consistent = shapes_consistent && first_shape == sublist.shape(0);
            if (!shapes_consistent)
               break;
         }
         TENSOR_CHECK(shapes_consistent,
                      printf("Inconsistent shapes in initializer list for %juD initializer tensor.\n",
                             static_cast<uintmax_t>(NumDims)));
      }

      static constexpr index_t numDims()
      {
         return NumDims;
      }

      constexpr index_t shape(index_t dim) const
      {
         TENSOR_DEBUG_ASSERT(dim < NumDims, printf("Dimension %ju out of bounds for %juD initializer tensor.\n",
                                                   static_cast<uintmax_t>(dim), static_cast<uintmax_t>(NumDims)));

         if (dim == 0)
         {
            return data.size();
         }
         else
         {
            return data.begin()->shape(dim - 1);
         }
      }

      constexpr index_t size() const
      {
         return data.size() * data.begin()->size();
      }

      constexpr const auto &operator[](index_t l) const
      {
         index_t s = data.size();
         index_t i = l % s;
         index_t j = l / s;
         return (*std::next(data.begin(), i))[j];
      }
   };

   /**
    * @brief Copies data from an InitializerTensor to a target tensor-like structure.
    *
    * @tparam T The value type of the initializer tensor.
    * @tparam NumDims The number of dimensions of the initializer tensor.
    * @tparam TensorLike The type of the target tensor-like structure.
    * @param target The target tensor-like structure to copy data into.
    * @param init The InitializerTensor to copy data from.
    */
   template <typename T, index_t NumDims, typename TensorLike>
   TENSOR_FUNC void fromInitializer(TensorLike &&target, InitializerTensor<T, NumDims> init)
   {
      TENSOR_REQUIRE_EQUAL_SHAPES(target, init);

      for (index_t i = 0; i < target.size(); ++i)
      {
         target[i] = init[i];
      }
   }
} // namespace tensor::details