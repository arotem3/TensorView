#pragma once
#include <memory>

#include "TensorView/Access/computeLinearIndex.hpp"
#include "TensorView/Macros.hpp"

namespace tensor::details
{
   template <typename TensorType>
   class TensorIterator;

   struct TensorEndSentinel
   {};

   template <typename TensorType>
   auto TensorBegin(TensorType &&tensor)
   {
      using view_t = decltype(tensor.view());
      return TensorIterator<view_t>(tensor.view(), 0);
   }

   template <typename TensorType, size_t NumDims>
   constexpr std::ptrdiff_t incrMultiIndex(std::array<index_t, NumDims> &indices, const TensorType &t)
   {
      for (index_t d = 0; d < NumDims; ++d)
      {
         if (++indices[d] < t.shape(d))
            return 0;
         indices[d] = 0;
      }
      return 1; // Reached the end of the multi-index
   }

   template <typename TensorType, size_t NumDims>
   constexpr std::ptrdiff_t incrMultiIndex(std::array<index_t, NumDims> &indices, const TensorType &t, std::ptrdiff_t n)
   {
      for (index_t d = 0; d < NumDims && n != 0; ++d)
      {
         const std::ptrdiff_t dim = t.shape(d);
         const std::ptrdiff_t sum = static_cast<std::ptrdiff_t>(indices[d]) + n;

         std::ptrdiff_t new_idx = sum % dim;
         if (new_idx < 0)
            new_idx += dim;

         n = (sum - new_idx) / dim;
         indices[d] = static_cast<index_t>(new_idx);
      }
      return n;
   }

   template <typename TensorType>
   class TensorIterator
   {
   public:
      using multi_index = std::array<index_t, TensorType::numDims()>;
      using iterator_category = std::random_access_iterator_tag;
      using difference_type = std::ptrdiff_t;
      using value_type = typename TensorType::value_type;
      using reference = decltype(std::declval<TensorType &>().at(std::declval<const multi_index &>()));
      using const_reference = decltype(std::declval<const TensorType &>().at(std::declval<const multi_index &>()));
      using pointer = std::conditional_t<std::is_lvalue_reference_v<reference>,
                                         std::add_pointer_t<std::remove_reference_t<reference>>, void>;
      using const_pointer = std::conditional_t<std::is_lvalue_reference_v<const_reference>,
                                               std::add_pointer_t<std::remove_reference_t<const_reference>>, void>;

   private:
      TensorType _tensor;
      multi_index _pos = {};
      bool _end = true;

   public:
      constexpr TensorIterator() = default;
      constexpr TensorIterator(const TensorIterator &) = default;
      constexpr TensorIterator(TensorIterator &&) = default;

      // Explicitly implement assignment to work even when TensorType has const elements
      // (and thus TensorType's assignment operators are deleted due to _is_mutable constraint)
      constexpr TensorIterator &operator=(const TensorIterator &other)
      {
         if (this != &other)
         {
            std::destroy_at(&_tensor);
            std::construct_at(&_tensor, other._tensor);
            _pos = other._pos;
            _end = other._end;
         }
         return *this;
      }

      constexpr TensorIterator &operator=(TensorIterator &&other) noexcept
      {
         if (this != &other)
         {
            std::destroy_at(&_tensor);
            std::construct_at(&_tensor, std::move(other._tensor));
            _pos = std::move(other._pos);
            _end = other._end;
         }
         return *this;
      }

      constexpr TensorIterator(TensorType tensor, difference_type pos = 0) : _tensor(tensor), _pos{}
      {
         _end = incrMultiIndex(_pos, _tensor, pos) != 0;
      }

      constexpr TensorIterator(TensorType tensor, multi_index pos, bool end) : _tensor(tensor), _pos(pos), _end(end) {}

      TENSOR_HOST_DEVICE const_reference operator*() const
      {
         return _tensor.at(_pos);
      }

      TENSOR_HOST_DEVICE reference operator*()
      {
         return _tensor.at(_pos);
      }

      TENSOR_HOST_DEVICE const_pointer operator->() const
         requires(!std::is_void_v<const_pointer>)
      {
         return &_tensor.at(_pos);
      }

      TENSOR_HOST_DEVICE pointer operator->()
         requires(!std::is_void_v<pointer>)
      {
         return &_tensor.at(_pos);
      }

      TENSOR_HOST_DEVICE const_reference operator[](difference_type n) const
      {
         multi_index pos_plus_n = _pos;
         incrMultiIndex(pos_plus_n, _tensor, n);
         return _tensor.at(pos_plus_n);
      }

      TENSOR_HOST_DEVICE reference operator[](difference_type n)
      {
         multi_index pos_plus_n = _pos;
         incrMultiIndex(pos_plus_n, _tensor, n);
         return _tensor.at(pos_plus_n);
      }

      TENSOR_FUNC TensorIterator &operator++()
      {
         _end = incrMultiIndex(_pos, _tensor) != 0;
         return *this;
      }

      TENSOR_FUNC TensorIterator operator++(int)
      {
         TensorIterator temp = *this;
         _end = incrMultiIndex(_pos, _tensor) != 0;
         return temp;
      }

      TENSOR_FUNC TensorIterator &operator--()
      {
         _end = incrMultiIndex(_pos, _tensor, -1) != 0;
         return *this;
      }

      TENSOR_FUNC TensorIterator operator--(int)
      {
         TensorIterator temp = *this;
         _end = incrMultiIndex(_pos, _tensor, -1) != 0;
         return temp;
      }

      TENSOR_FUNC TensorIterator operator+(difference_type n) const
      {
         multi_index pos_plus_n = _pos;
         bool end = incrMultiIndex(pos_plus_n, _tensor, n) != 0;
         return TensorIterator(_tensor, pos_plus_n, end);
      }

      TENSOR_FUNC TensorIterator &operator+=(difference_type n)
      {
         _end = incrMultiIndex(_pos, _tensor, n) != 0;
         return *this;
      }

      TENSOR_FUNC TensorIterator operator-(difference_type n) const
      {
         multi_index pos_minus_n = _pos;
         bool end = incrMultiIndex(pos_minus_n, _tensor, -n) != 0;
         return TensorIterator(_tensor, pos_minus_n, end);
      }

      TENSOR_FUNC TensorIterator &operator-=(difference_type n)
      {
         _end = incrMultiIndex(_pos, _tensor, -n) != 0;
         return *this;
      }

      TENSOR_FUNC difference_type operator-(const TensorIterator &other) const
      {
         return _pos - other._pos;
      }

      TENSOR_FUNC bool operator==(const TensorIterator &other) const
      {
         return (_end == other._end) && (_pos == other._pos);
      }

      TENSOR_FUNC bool operator!=(const TensorIterator &other) const
      {
         return !(*this == other);
      }

      TENSOR_FUNC bool operator<(const TensorIterator &other) const
      {
         bool less = std::lexicographical_compare(_pos.begin(), _pos.end(), other._pos.begin(), other._pos.end());
         return (_end == other._end) ? less : _end < other._end;
      }

      TENSOR_FUNC bool operator<=(const TensorIterator &other) const
      {
         return (*this < other) || (*this == other);
      }

      TENSOR_FUNC bool operator>(const TensorIterator &other) const
      {
         return !(*this <= other);
      }

      TENSOR_FUNC bool operator>=(const TensorIterator &other) const
      {
         return !(*this < other);
      }

      friend TENSOR_FUNC TensorIterator operator+(difference_type n, const TensorIterator &it)
      {
         multi_index pos_plus_n = it._pos;
         bool end = incrMultiIndex(pos_plus_n, it._tensor, n) != 0;
         return TensorIterator(it._tensor, pos_plus_n, end);
      }

      friend TENSOR_FUNC bool operator==(const TensorIterator &it, TensorEndSentinel)
      {
         return it._end;
      }

      friend TENSOR_FUNC bool operator==(TensorEndSentinel, const TensorIterator &it)
      {
         return it._end;
      }

      friend TENSOR_FUNC bool operator!=(const TensorIterator &it, TensorEndSentinel)
      {
         return !it._end;
      }

      friend TENSOR_FUNC bool operator!=(TensorEndSentinel, const TensorIterator &it)
      {
         return !it._end;
      }
   };
} // namespace tensor::details