#ifndef __TENSOR_VIEW_BASE_TENSOR_HPP__
#define __TENSOR_VIEW_BASE_TENSOR_HPP__

#include "tensorview_config.hpp"
#include "errors.hpp"
#include "span.hpp"

namespace tensor
{
  template <typename scalar, size_t Rank>
  class SubView;
} // namespace tensor

namespace tensor::details
{
  template <typename Shape, typename Container>
  class BaseTensor
  {
  public:
    using value_type = typename Container::value_type;
    using reference = typename Container::reference;
    using const_reference = typename Container::const_reference;
    using pointer = typename Container::pointer;
    using const_pointer = typename Container::const_pointer;

    using shape_type = Shape;
    using container_type = Container;

    template <typename ViewType>
    class Iterator
    {
    public:
      using iterator_category = std::random_access_iterator_tag;
      using value_type = typename ViewType::value_type;
      using difference_type = std::ptrdiff_t;
      using pointer = value_type *;
      using reference = value_type &;

      // sentinel
      TENSOR_FUNC Iterator()
          : initialized(false), _pos(0) {}

      TENSOR_FUNC Iterator(Shape shape_, ViewType &&view_, index_t pos)
          : initialized(true), _pos(pos), _shape(shape_), _view(view_) {}

      TENSOR_FUNC reference operator*()
      {
#ifdef TENSOR_DEBUG
        if (!initialized)
          tensor_bad_memory_access();
#endif
        return _view[_shape[_pos]];
      }

      TENSOR_FUNC pointer operator->()
      {
#ifdef TENSOR_DEBUG
        if (!initialized)
          tensor_bad_memory_access();
#endif
        return &_view[_shape[_pos]];
      }

      TENSOR_FUNC reference operator[](difference_type n)
      {
#ifdef TENSOR_DEBUG
        if (!initialized)
          tensor_bad_memory_access();
#endif
        return _view[_shape[_pos + n]];
      }

      TENSOR_FUNC Iterator &operator++()
      {
#ifdef TENSOR_DEBUG
        if (!initialized)
          tensor_bad_memory_access();
#endif
        ++_pos;
        return *this;
      }

      TENSOR_FUNC Iterator operator++(int)
      {
#ifdef TENSOR_DEBUG
        if (!initialized)
          tensor_bad_memory_access();
#endif
        Iterator tmp = *this;
        ++_pos;
        return tmp;
      }

      TENSOR_FUNC Iterator &operator--()
      {
#ifdef TENSOR_DEBUG
        if (!initialized)
          tensor_bad_memory_access();
#endif
        --_pos;
        return *this;
      }

      TENSOR_FUNC Iterator operator--(int)
      {
#ifdef TENSOR_DEBUG
        if (!initialized)
          tensor_bad_memory_access();
#endif
        Iterator tmp = *this;
        --_pos;
        return tmp;
      }

      TENSOR_FUNC Iterator operator+(difference_type n) const
      {
#ifdef TENSOR_DEBUG
        if (!initialized)
          tensor_bad_memory_access();
#endif
        return Iterator(_shape, _view, _pos + n);
      }

      TENSOR_FUNC Iterator operator-(difference_type n) const
      {
#ifdef TENSOR_DEBUG
        if (!initialized)
          tensor_bad_memory_access();
#endif
        return Iterator(_shape, _view, _pos - n);
      }

      TENSOR_FUNC Iterator &operator+=(difference_type n)
      {
#ifdef TENSOR_DEBUG
        if (!initialized)
          tensor_bad_memory_access();
#endif
        _pos += n;
        return *this;
      }

      TENSOR_FUNC Iterator &operator-=(difference_type n)
      {
#ifdef TENSOR_DEBUG
        if (!initialized)
          tensor_bad_memory_access();
#endif
        _pos -= n;
        return *this;
      }

      TENSOR_FUNC difference_type operator-(const Iterator &other) const
      {
#ifdef TENSOR_DEBUG
        if (!initialized || !other.initialized)
          tensor_bad_memory_access();
#endif
        return _pos - other._pos;
      }

      TENSOR_FUNC bool operator==(const Iterator &other) const
      {
        return initialized && other.initialized && (_pos == other._pos);
      }

      TENSOR_FUNC bool operator!=(const Iterator &other) const
      {
        return (not initialized) or (not other.initialized) or (_pos != other._pos);
      }

      TENSOR_FUNC bool operator<(const Iterator &other) const
      {
        return (!initialized || (other.initialized && _pos < other._pos));
      }

      TENSOR_FUNC bool operator>(const Iterator &other) const
      {
        return !other.initialized || (initialized && _pos > other._pos);
      }

      TENSOR_FUNC bool operator<=(const Iterator &other) const
      {
        return (!initialized || (other.initialized && _pos <= other._pos));
      }

      TENSOR_FUNC bool operator>=(const Iterator &other) const
      {
        return !other.initialized || (initialized && _pos >= other._pos);
      }

    private:
      bool initialized;
      index_t _pos;
      Shape _shape;
      ViewType _view;
    };

    using iterator = Iterator<decltype(wrap_view(std::declval<Container &>()))>;
    using const_iterator = Iterator<decltype(wrap_view(std::declval<const Container &>()))>;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    BaseTensor() = default;
    ~BaseTensor() = default;

    explicit TENSOR_FUNC BaseTensor(Shape shape_, Container container_) : _shape(shape_), container(container_) {}

    /// @brief shallow copy.
    BaseTensor(const BaseTensor &) = default;
    BaseTensor &operator=(const BaseTensor &) = default;

    /// @brief move
    BaseTensor(BaseTensor &&) = default;
    BaseTensor &operator=(BaseTensor &&) = default;

    /// @brief returns the order of the tensor, i.e. the number of dimensions
    /// of the tensor.
    static constexpr index_t order()
    {
      return Shape::order();
    }

    /**
     * @brief high dimensional read/write access.
     *
     * @tparam Indices `span` or convertible to `index_t`
     * @param indices indices
     * @return if all indices are integers, the returns a reference to the
     * data at the index. If any indices are `span` then returns a TensorView
     * of the range specified by the spans.
     */
    template <typename... Indices>
    TENSOR_FUNC decltype(auto) at(Indices... indices)
    {
      return subview(_shape(std::forward<Indices>(indices)...));
    }

    /**
     * @brief high dimensional read/write access.
     *
     * @tparam Indices `span` or convertible to `index_t`
     * @param indices indices
     * @return if all indices are integers, the returns a reference to the
     * data at the index. If any indices are `span` then returns a TensorView
     * of the range specified by the spans.
     */
    template <typename... Indices>
    TENSOR_FUNC decltype(auto) at(Indices... indices) const
    {
      return subview(_shape(std::forward<Indices>(indices)...));
    }

    /**
     * @brief high dimensional read/write access.
     *
     * @tparam Indices `span` or convertible to `index_t`
     * @param indices indices
     * @return if all indices are integers, the returns a reference to the
     * data at the index. If any indices are `span` then returns a TensorView
     * of the range specified by the spans.
     */
    template <typename... Indices>
    TENSOR_FUNC decltype(auto) operator()(Indices... indices)
    {
      return subview(_shape(std::forward<Indices>(indices)...));
    }

    /**
     * @brief high dimensional read/write access.
     *
     * @tparam Indices `span` or convertible to `index_t`
     * @param indices indices
     * @return if all indices are integers, the returns a reference to the
     * data at the index. If any indices are `span` then returns a TensorView
     * of the range specified by the spans.
     */
    template <typename... Indices>
    TENSOR_FUNC decltype(auto) operator()(Indices... indices) const
    {
      return subview(_shape(std::forward<Indices>(indices)...));
    }

    /// @brief linear index access.
    TENSOR_FUNC reference operator[](index_t index)
    {
      return container[_shape[index]];
    }

    /// @brief linear index access.
    TENSOR_FUNC const_reference operator[](index_t index) const
    {
      return container[_shape[index]];
    }

    /// @brief returns pointer to start of tensor.
    TENSOR_FUNC iterator begin()
    {
      return iterator(_shape, wrap_view<Container>(container), 0);
    }

    /// @brief returns pointer to start of tensor.
    TENSOR_FUNC const_iterator begin() const
    {
      return const_iterator(_shape, wrap_view<const Container>(container), 0);
    }

    /// @brief returns pointer to the element following the last element of tensor.
    TENSOR_FUNC iterator end()
    {
      return iterator(_shape, wrap_view<Container>(container), size());
    }

    /// @brief returns pointer to the element following the last element of tensor.
    TENSOR_FUNC const_iterator end() const
    {
      return const_iterator(_shape, wrap_view<const Container>(container), size());
    }

    /// @brief returns reverse iterator to the end of the tensor.
    TENSOR_FUNC reverse_iterator rbegin()
    {
      return reverse_iterator(end());
    }

    /// @brief returns reverse iterator to the end of the tensor.
    TENSOR_FUNC const_reverse_iterator rbegin() const
    {
      return const_reverse_iterator(end());
    }

    /// @brief returns reverse iterator to the start of the tensor.
    TENSOR_FUNC reverse_iterator rend()
    {
      return reverse_iterator(begin());
    }

    /// @brief returns reverse iterator to the start of the tensor.
    TENSOR_FUNC const_reverse_iterator rend() const
    {
      return const_reverse_iterator(begin());
    }

    /// @brief returns total number of elements in the tensor.
    TENSOR_FUNC index_t size() const
    {
      return _shape.size();
    }

    /// @brief returns the size of the tensor along dimension d.
    TENSOR_FUNC index_t shape(index_t d) const
    {
      return _shape.shape(d);
    }

  protected:
    Shape _shape;
    Container container;

    TENSOR_FUNC reference subview(index_t index)
    {
      return container[index];
    }

    TENSOR_FUNC const_reference subview(index_t index) const
    {
      return container[index];
    }

    template <size_t N>
    TENSOR_FUNC auto subview(const std::array<span, N> &spans)
    {
      return SubView<value_type, N>(wrap_view(container), spans);
    }

    template <size_t N>
    TENSOR_FUNC auto subview(const std::array<span, N> &spans) const
    {
      return SubView<const value_type, N>(wrap_view(container), spans);
    }

    TENSOR_FUNC auto subview(const span &s)
    {
      return SubView<value_type, 1>(wrap_view(container), s);
    }

    TENSOR_FUNC auto subview(const span &s) const
    {
      return SubView<const value_type, 1>(wrap_view(container), s);
    }
  };

} // namespace tensor::details

#endif
