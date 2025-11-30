#pragma once
#include "TensorView/Access/SimplifyIndex.hpp"
#include "TensorView/Access/Span.hpp"
#include "TensorView/Macros.hpp"
#include "TensorView/Shapes/LinearOrder.hpp"
#include "TensorView/Shapes/ShapeTraits.hpp"

namespace tensor::details
{
   template <index_t NumDims, LinearOrder Order>
   class StandardShape
   {
      static_assert(NumDims > 0, "StandardShape must have a non-zero number of dimensions.");

   public:
      static constexpr LinearOrder linear_order = Order;

   private:
      std::array<index_t, NumDims> _shape;

      friend struct ShapeTraits<StandardShape<NumDims, Order>>;

   public:
      constexpr StandardShape() = default;
      constexpr StandardShape(StandardShape &&) = default;
      constexpr StandardShape &operator=(StandardShape &&) = default;
      constexpr StandardShape(const StandardShape &) = default;
      constexpr StandardShape &operator=(const StandardShape &) = default;

      template <IndexLike... Shape>
      TENSOR_FUNC explicit StandardShape(Shape... shape_) : _shape{static_cast<index_t>(shape_)...}
      {
         static_assert(sizeof...(shape_) <= NumDims,
                       "Too many dimensions specified for StandardShape of given number of dimensions.");

         TENSOR_DEBUG_ASSERT(((std::is_unsigned_v<Shape> || (shape_ >= 0)) && ... && true),
                             printf("Shape dimensions must be non-negative.\n"));

         // fill in the rest of the shape with 1s
         for (index_t i = sizeof...(shape_); i < NumDims; ++i)
            _shape[i] = 1;
      }

      static constexpr index_t numDims()
      {
         return NumDims;
      }

      /**
       * @brief Is the shape F-contiguous
       */
      static constexpr bool contiguous()
      {
         return Order == LinearOrder::F;
      }

      /**
       * @brief Returns the total number of elements described by the shape.
       */
      constexpr index_t size() const
      {
         index_t len = 1;
         for (index_t s : _shape)
            len *= s;
         return len;
      }

      /**
       * @brief Returns the total range in memory covered by the shape.
       */
      constexpr index_t extent() const
      {
         return size() + offset();
      }

      /**
       * @brief returns the offset in memory of the shape from the base pointer.
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
         TENSOR_DEBUG_ASSERT(dim < NumDims, printf("Dimension %ju is out of range for shape with %ju dimensions.\n",
                                                   static_cast<uintmax_t>(dim), static_cast<uintmax_t>(NumDims)));
         return _shape[dim];
      }

      /**
       * @brief Computes the linear index corresponding to the provided multi-dimensional indices.
       */
      template <typename... Indices>
      TENSOR_FUNC auto operator()(Indices... indices) const
      {
         static_assert(sizeof...(Indices) == NumDims, "wrong number of indices.");
         constexpr index_t start = (Order == LinearOrder::F) ? 0 : NumDims - 1;
         return offset() + computeIndex<start>(std::forward_as_tuple(indices...));
      }

      /**
       * @brief linear index operator.
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
            index_t l = offset();
            auto strides = details::CStrides(_shape);

            for (index_t d = 0; d < NumDims; ++d)
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
         static_assert(Dim < NumDims, "Dimension out of range in computeIndex.");

         decltype(auto) index = details::simplifyIndex(std::get<Dim>(indices), Dim, _shape[Dim]);

         if constexpr (Order == LinearOrder::F)
         {
            if constexpr (Dim + 1 < NumDims)
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

   template <index_t NumDims, LinearOrder Order>
   struct ShapeTraits<StandardShape<NumDims, Order>>
   {
      using shape_type = StandardShape<NumDims, Order>;

      /**
       * @brief returns the number of dimensions of the shape.
       */
      static constexpr index_t numDims()
      {
         return NumDims;
      }

      /**
       * @brief Are all valid instances of this shape contiguous in memory with respect to LinearOrder::F?
       */
      static constexpr bool contiguous()
      {
         return Order == LinearOrder::F;
      }

      template <index_t N>
      static constexpr shape_type from(const StandardShape<N, Order> &other)
         requires(N <= NumDims)
      {
         if constexpr (N == NumDims)
            return other;
         else
         {
            shape_type shape;
            for (index_t i = 0; i < NumDims; ++i)
            {
               shape._shape[i] = (i < N) ? other.shape(i) : 1;
            }
            return shape;
         }
      }

      template <typename ShapeLike>
      static constexpr shape_type makeLike(const ShapeLike &other)
      {
         shape_type shape;
         constexpr index_t other_num_dims = ShapeTraits<ShapeLike>::numDims();
         static_assert(other_num_dims <= NumDims,
                       "Cannot make StandardShape like ShapeLike with more dimensions than NumDims.");

         for (index_t i = 0; i < NumDims; ++i)
         {
            shape._shape[i] = (i < other_num_dims) ? other.shape(i) : 1;
         }

         return shape;
      }
   };

   template <typename T>
   struct IsStandardShape : std::false_type
   {
   };

   template <index_t NumDims, LinearOrder Order>
   struct IsStandardShape<StandardShape<NumDims, Order>> : std::true_type
   {
   };
} // namespace tensor::details
