#pragma once
#include "TensorView/Access/SimplifyIndex.hpp"
#include "TensorView/Access/Span.hpp"
#include "TensorView/Macros.hpp"
#include "TensorView/Shapes/LinearOrder.hpp"
#include "TensorView/Shapes/ShapeTraits.hpp"

namespace tensor::details
{
   /**
    * @brief Represents a tensor shape with static (compile-time) dimensions.
    */
   template <LinearOrder Order, index_t... Dims>
   class StaticShape
   {
   public:
      static constexpr LinearOrder linear_order = Order;

   private:
      friend struct ShapeTraits<StaticShape<Order, Dims...>>;

   public:
      constexpr StaticShape() = default;
      constexpr StaticShape(StaticShape &&) = default;
      constexpr StaticShape &operator=(StaticShape &&) = default;
      constexpr StaticShape(const StaticShape &) = default;
      constexpr StaticShape &operator=(const StaticShape &) = default;
      constexpr ~StaticShape() = default;

      static constexpr index_t numDims()
      {
         return sizeof...(Dims);
      }

      /**
       * @brief Is the shape F-contiguous?
       */
      static constexpr bool contiguous()
      {
         return Order == LinearOrder::F;
      }

      /**
       * @brief Returns the total number of elements described by the shape.
       */
      static constexpr index_t size()
      {
         return (1 * ... * Dims);
      }

      /**
       * @brief Returns the total range in memory covered by the shape.
       */
      static constexpr index_t extent()
      {
         return size();
      }

      /**
       * @brief Returns the offset of the shape (always 0 for StaticShape).
       */
      constexpr index_t offset() const
      {
         return 0;
      }

      /**
       * @brief Is the shape logically empty (i.e., has zero elements)?
       */
      constexpr bool empty() const
      {
         return size() == 0;
      }

      /**
       * @brief Returns the size of the specified dimension.
       */
      TENSOR_FUNC index_t shape(index_t dim) const
      {
         TENSOR_DEBUG_ASSERT(dim < numDims(), printf("Dimension %ju is out of range for shape with %ju dimensions.\n",
                                                     static_cast<uintmax_t>(dim), static_cast<uintmax_t>(numDims())));

         constexpr index_t _shape[] = {Dims...};
         return _shape[dim];
      }

      /**
       * @brief Computes the linear index corresponding to the provided multi-dimensional indices.
       */
      template <typename... Indices>
      TENSOR_FUNC auto operator()(Indices... indices) const
      {
         static_assert(sizeof...(Indices) == numDims(), "wrong number of indices.");
         constexpr index_t start = (Order == LinearOrder::F) ? 0 : numDims() - 1;
         return computeIndex<start>(std::forward_as_tuple(indices...));
      }

      /**
       * @brief Identity mapping for linear indices.
       */
      TENSOR_FUNC index_t operator[](index_t index) const
      {
         TENSOR_DEBUG_ASSERT(index < size(), printf("Linear index = %ju is out of range for tensor with size %ju.\n",
                                                    static_cast<uintmax_t>(index), static_cast<uintmax_t>(size())));
         if constexpr (Order == LinearOrder::F)
         {
            return offset() + index;
         }
         else // C order: need to convert linear index to Fortran order
         {
            constexpr index_t _shape[] = {Dims...};
            constexpr auto strides = details::CStrides({Dims...});

            index_t l = offset();

            for (index_t d = 0; d < numDims(); ++d)
            {
               l += strides[d] * (index % _shape[d]);
               index /= _shape[d];
            }

            return l;
         }
      }

   private:
      template <index_t Dim, typename IndexTuple>
      TENSOR_FUNC auto computeIndex(IndexTuple &&indices) const
      {
         static_assert(Dim < numDims(), "Dimension out of range in computeIndex.");

         constexpr index_t _shape[] = {Dims...};
         decltype(auto) index = details::simplifyIndex(std::get<Dim>(indices), Dim, _shape[Dim]);

         if constexpr (Order == LinearOrder::F)
         {
            if constexpr (Dim + 1 < numDims())
               return index + _shape[Dim] * computeIndex<Dim + 1>(std::forward<IndexTuple>(indices));
            else
               return index;
         }
         else // C order
         {
            if constexpr (Dim > 0)
               return _shape[Dim] * computeIndex<Dim - 1>(std::forward<IndexTuple>(indices)) + index;
            else
               return index;
         }
      }
   };

   template <LinearOrder Order, index_t... Dims>
   struct ShapeTraits<StaticShape<Order, Dims...>>
   {
      using shape_type = StaticShape<Order, Dims...>;

      static constexpr index_t numDims()
      {
         return sizeof...(Dims);
      }

      /**
       * @brief Are all instances of this shape type F-contiguous?
       */
      static constexpr bool contiguous()
      {
         return Order == LinearOrder::F;
      }

      template <typename ShapeType>
      static TENSOR_HOST_DEVICE shape_type from([[maybe_unused]] const ShapeType &other)
         requires(ShapeType::numDims() <= sizeof...(Dims))
      {
         shape_type shape;

#ifndef NDEBUG
         constexpr index_t _shape[] = {Dims...};
         for (index_t i = 0; i < sizeof...(Dims); ++i)
         {
            index_t dim = (i < ShapeType::numDims()) ? other.shape(i) : 1;
            TENSOR_DEBUG_ASSERT(
                dim == _shape[i],
                printf("Cannot convert to StaticShape: expected %ju but got %ju at dimension %ju.\n",
                       static_cast<uintmax_t>(_shape[i]), static_cast<uintmax_t>(dim), static_cast<uintmax_t>(i)));
         }
#endif

         return shape;
      }
   };

   template <typename T>
   struct IsStaticShape : std::false_type
   {
   };

   template <LinearOrder Order, index_t... Dims>
   struct IsStaticShape<StaticShape<Order, Dims...>> : std::true_type
   {
   };
} // namespace tensor::details
