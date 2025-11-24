#pragma once
#include "TensorView/Containers/ContainerTraits.hpp"
#include "TensorView/Macros.hpp"
#include "TensorView/Shapes/ShapeTraits.hpp"

namespace tensor::details
{
   /**
    * @brief Iterator for tensors.
    * These are not persistent references to the container, they are raw views.
    */
   template <typename Shape, typename Container>
   class TensorIterator
   {
   public:
      using container_type = Container;
      using shape_type = Shape;

      using ct = ContainerTraits<Container>;
      using st = ShapeTraits<Shape>;

   private:
      static constexpr bool _contiguous = st::contiguous();
      static constexpr bool _mutable = ct::mutableElements();

   public:
      using iterator_category =
          std::conditional_t<_contiguous, std::contiguous_iterator_tag, std::random_access_iterator_tag>;

      using value_type = typename ct::value_type;
      using difference_type = std::ptrdiff_t;
      using pointer = std::conditional_t<_mutable || _contiguous, value_type *, void>;
      using reference = std::conditional_t<_mutable || _contiguous, value_type &, void>;

   private:
      shape_type _shape;
      mutable container_type _container;
      difference_type _pos;

   public:
      constexpr TensorIterator() = default;
      constexpr TensorIterator(const TensorIterator &) = default;
      constexpr TensorIterator &operator=(const TensorIterator &) = default;

      TENSOR_FUNC TensorIterator(const shape_type &shape_, container_type &&container_, difference_type pos)
          : _shape(shape_), _container(std::forward<container_type>(container_)), _pos(pos)
      {
      }

      TENSOR_HOST_DEVICE inline decltype(auto) operator*() const
      {
         return _container[_shape[_pos]];
      }

      TENSOR_HOST_DEVICE inline pointer operator->() const
         requires(_mutable || _contiguous)
      {
         return &_container[_shape[_pos]];
      }

      TENSOR_HOST_DEVICE inline decltype(auto) operator[](difference_type n) const
      {
         return _container[_shape[_pos + n]];
      }

      TENSOR_FUNC TensorIterator &operator++()
      {
         ++_pos;
         return *this;
      }

      TENSOR_FUNC TensorIterator operator++(int)
      {
         TensorIterator tmp = *this;
         ++_pos;
         return tmp;
      }

      TENSOR_FUNC TensorIterator &operator--()
      {
         --_pos;
         return *this;
      }

      TENSOR_FUNC TensorIterator operator--(int)
      {
         TensorIterator tmp = *this;
         --_pos;
         return tmp;
      }

      TENSOR_FUNC TensorIterator operator+(difference_type n) const
      {
         return TensorIterator(_shape, Container(_container), _pos + n);
      }

      TENSOR_FUNC TensorIterator operator-(difference_type n) const
      {
         return TensorIterator(_shape, Container(_container), _pos - n);
      }

      TENSOR_FUNC TensorIterator &operator+=(difference_type n)
      {
         _pos += n;
         return *this;
      }

      TENSOR_FUNC TensorIterator &operator-=(difference_type n)
      {
         _pos -= n;
         return *this;
      }

      TENSOR_FUNC difference_type operator-(const TensorIterator &other) const
      {
         return _pos - other._pos;
      }

      TENSOR_FUNC bool operator==(const TensorIterator &other) const
      {
         return _pos == other._pos;
      }

      TENSOR_FUNC bool operator!=(const TensorIterator &other) const
      {
         return _pos != other._pos;
      }

      TENSOR_FUNC bool operator<(const TensorIterator &other) const
      {
         return _pos < other._pos;
      }

      TENSOR_FUNC bool operator>(const TensorIterator &other) const
      {
         return _pos > other._pos;
      }

      TENSOR_FUNC bool operator<=(const TensorIterator &other) const
      {
         return _pos <= other._pos;
      }

      TENSOR_FUNC bool operator>=(const TensorIterator &other) const
      {
         return _pos >= other._pos;
      }

      friend TENSOR_FUNC TensorIterator operator+(difference_type n, const TensorIterator &it)
      {
         return it + n;
      }
   };
} // namespace tensor::details
