#pragma once
#include "TensorView/Access/SimplifyIndex.hpp"
#include "TensorView/Access/Span.hpp"
#include "TensorView/Macros.hpp"
#include "TensorView/Shapes/LinearOrder.hpp"
#include "TensorView/Shapes/ShapeTraits.hpp"

namespace tensor::details
{
   template <index_t numDims, LinearOrder Order>
   class StandardShape;

   template <LinearOrder Order, index_t... Dims>
   class StaticShape;

   template <index_t NumDims>
   class StridedShape
   {
      static_assert(NumDims > 0, "StridedShape must have a non-zero number of dimensions.");

   private:
      std::array<index_t, NumDims> _shape;
      std::array<index_t, NumDims> _strides;
      index_t _offset;

      friend struct ShapeTraits<StridedShape<NumDims>>;

   public:
      constexpr StridedShape() = default;
      constexpr StridedShape(const StridedShape &) = default;
      constexpr StridedShape(StridedShape &&) = default;
      constexpr StridedShape &operator=(const StridedShape &) = default;
      constexpr StridedShape &operator=(StridedShape &&) = default;

      TENSOR_HOST_DEVICE StridedShape(const MultiSpan<NumDims> &spans)
      {
         _offset = tensor::details::offset(spans);

         for (index_t d = 0; d < NumDims; ++d)
         {
            _shape[d] = spans[d].size();
            _strides[d] = spans[d].stride;

            TENSOR_CHECK(_strides[d] >= 1, printf("Strides must be >= 1, but on dimensions %ju got %ju.\n",
                                                  static_cast<uintmax_t>(d), static_cast<uintmax_t>(_strides[d])));
         }
      }

      TENSOR_HOST_DEVICE StridedShape(const Span &s)
         requires(NumDims == 1)
      {
         _offset = s.begin;
         _shape[0] = s.size();
         _strides[0] = s.stride;

         TENSOR_CHECK(_strides[0] >= 1,
                      printf("Stride for dimension 0 must be >= 1, got %ju.\n", static_cast<uintmax_t>(_strides[0])));
      }

      TENSOR_FUNC StridedShape(std::array<index_t, NumDims> &&shape_, std::array<index_t, NumDims> &&strides_,
                               index_t offset = 0)
          : _shape(std::move(shape_)), _strides(std::move(strides_)), _offset(offset)
      {
         for (index_t d = 0; d < NumDims; ++d)
         {
            TENSOR_CHECK(_strides[d] >= 1, printf("Stride for dimension %ju must be >= 1, got %ju.\n",
                                                  static_cast<uintmax_t>(d), static_cast<uintmax_t>(_strides[d])));
         }
      }

      static constexpr index_t numDims()
      {
         return NumDims;
      }

      /**
       * @brief Is the shape F-contiguous?
       */
      constexpr bool contiguous() const
      {
         bool c = true;
         index_t s = 1;

         for (index_t k = 0; k < NumDims; ++k)
         {
            c = c && (s == _strides[k]);
            s *= _shape[k];
         }

         return c;
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
      TENSOR_FUNC index_t extent() const
      {
         return 1 + operator[](size() - 1) - _offset;
      }

      constexpr index_t offset() const
      {
         return _offset;
      }

      constexpr bool empty() const
      {
         return size() == 0;
      }

      TENSOR_FUNC index_t shape(index_t dim) const
      {
         TENSOR_DEBUG_ASSERT(dim < NumDims, printf("Dimension %ju is out of range for shape with %ju dimensions.\n",
                                                   static_cast<uintmax_t>(dim), static_cast<uintmax_t>(NumDims)));
         return _shape[dim];
      }

      TENSOR_FUNC index_t stride(index_t dim) const
      {
         TENSOR_DEBUG_ASSERT(dim < NumDims, printf("Dimension %ju is out of range for shape with %ju dimensions.\n",
                                                   static_cast<uintmax_t>(dim), static_cast<uintmax_t>(NumDims)));
         return _strides[dim];
      }

      template <typename... Indices>
      TENSOR_FUNC auto operator()(Indices... indices) const
      {
         static_assert(sizeof...(Indices) == NumDims, "wrong number of indices.");
         return _offset + computeIndex<0>(std::forward<Indices>(indices)...);
      }

      TENSOR_FUNC index_t operator[](index_t index) const
      {
         TENSOR_DEBUG_ASSERT(index < size(), printf("Linear index = %ju is out of range for tensor with size %ju.\n",
                                                    static_cast<uintmax_t>(index), static_cast<uintmax_t>(size())));

         index_t l = _offset;
         for (index_t d = 0; d < NumDims; ++d)
         {
            l += _strides[d] * (index % _shape[d]);
            index /= _shape[d];
         }
         return l;
      }

   private:
      template <index_t Dim, typename Index, typename... Indices>
      TENSOR_FUNC auto computeIndex(Index i, Indices... indices) const
      {
         static_assert(Dim < NumDims, "Dimension out of range in computeIndex.");

         decltype(auto) index = details::simplifyIndex(i, Dim, _shape[Dim]);

         if constexpr (Dim + 1 < NumDims)
            return _strides[Dim] * index + computeIndex<Dim + 1>(std::forward<Indices>(indices)...);
         else
            return _strides[Dim] * index;
      }
   };

   template <index_t NumDims>
   struct ShapeTraits<StridedShape<NumDims>>
   {
      using shape_type = StridedShape<NumDims>;

      /**
       * @brief returns the number of dimensions of the shape.
       */
      static constexpr index_t numDims()
      {
         return NumDims;
      }

      /**
       * @brief Are All valid instances of this shape F-contiguous in memory?
       */
      static constexpr bool contiguous()
      {
         return false;
      }

      template <index_t N, LinearOrder Order>
      static constexpr shape_type from(const StandardShape<N, Order> &other)
         requires(N <= NumDims)
      {
         shape_type shape;

         if constexpr (Order == LinearOrder::F)
         {
            index_t stride = 1;
            for (index_t d = 0; d < NumDims; ++d)
            {
               shape._shape[d] = (d < N) ? other.shape(d) : 1;
               shape._strides[d] = stride;
               stride *= shape._shape[d];
            }
         }
         else // C order
         {
            index_t stride = 1;
            for (index_t d = NumDims; d-- > 0;)
            {
               shape._shape[d] = (d < N) ? other.shape(d) : 1;
               shape._strides[d] = stride;
               stride *= shape._shape[d];
            }
         }

         shape._offset = other.offset();
         return shape;
      }

      template <index_t N>
      static TENSOR_FUNC shape_type from(const StridedShape<N> &other)
         requires(N <= NumDims)
      {
         if constexpr (N == NumDims)
            return other;

         shape_type shape;
         for (index_t i = 0; i < NumDims; ++i)
         {
            shape._shape[i] = (i < N) ? other.shape(i) : 1;
            shape._strides[i] = (i < N) ? other._strides[i] : 1;
         }
         shape._offset = other.offset();
         return shape;
      }

      template <LinearOrder Order, index_t... Dims>
      static constexpr shape_type from(const StaticShape<Order, Dims...> &)
         requires(sizeof...(Dims) <= NumDims)
      {
         return from(StandardShape<sizeof...(Dims), Order>(Dims...));
      }
   };
} // namespace tensor::details
