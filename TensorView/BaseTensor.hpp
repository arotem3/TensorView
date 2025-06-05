#ifndef __TENSOR_VIEW_BASE_TENSOR_HPP__
#define __TENSOR_VIEW_BASE_TENSOR_HPP__

#include "tensorview_config.hpp"
#include "errors.hpp"
#include "ContainerTraits.hpp"
#include "span.hpp"
#include "ViewContainer.hpp"

namespace tensor
{
  template <size_t Rank, typename Container>
  class SubView;
} // namespace tensor

namespace tensor::details
{
  template <size_t Rank, typename Container>
  TENSOR_HOST_DEVICE inline auto make_subview(const Container &container, const std::array<span, Rank> &spans)
  {
    using shape_type = details::StridedShape<Rank>;
    using ct = container_traits<Container>;
    using container_type = typename ct::const_view_type;

    return SubView<Rank, container_type>(shape_type(spans), ct::make_view(container, details::offset(spans)));
  }

  template<size_t Rank, typename Container>
  TENSOR_HOST_DEVICE inline auto make_subview(Container &container, const std::array<span, Rank> &spans)
  {
    using shape_type = details::StridedShape<Rank>;
    using ct = container_traits<Container>;
    using container_type = typename ct::view_type;

    return SubView<Rank, container_type>(shape_type(spans), ct::make_view(container, details::offset(spans)));
  }

  template <typename Container>
  TENSOR_HOST_DEVICE inline auto make_subview(const Container &container, const span &s)
  {
    using shape_type = details::StridedShape<1>;
    using ct = container_traits<Container>;
    using container_type = typename ct::const_view_type;

    return SubView<1, container_type>(shape_type(s), ct::make_view(container, s.begin));
  }

  template <typename Container>
  TENSOR_HOST_DEVICE inline auto make_subview(Container &container, const span &s)
  {
    using shape_type = details::StridedShape<1>;
    using ct = container_traits<Container>;
    using container_type = typename ct::view_type;

    return SubView<1, container_type>(shape_type(s), ct::make_view(container, s.begin));
  }

  template <typename Shape, typename Container>
  class TensorIterator
  {
  private:
    static constexpr bool _contiguous = Shape::is_contiguous() && container_traits<Container>::is_contiguous;
    static constexpr bool _mutable = container_traits<Container>::is_mutable;

  public:
    using container_type = Container;
    using shape_type = Shape;

    using iterator_category = std::conditional_t<_contiguous, std::contiguous_iterator_tag, std::random_access_iterator_tag>;
    using value_type = typename Container::value_type;
    using difference_type = std::ptrdiff_t;
    using pointer = std::conditional_t<_mutable || _contiguous, value_type *, void>;
    using reference = std::conditional_t<_mutable || _contiguous, value_type &, void>;

    TENSOR_FUNC TensorIterator() = default;
    TENSOR_FUNC TensorIterator(const TensorIterator &) = default;
    TENSOR_FUNC TensorIterator &operator=(const TensorIterator &) = default;

    TENSOR_FUNC TensorIterator(const Shape &shape_, Container &&container_, difference_type pos)
        : _shape(shape_), _container(std::forward<Container>(container_)), _pos(pos) {}

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
      return TensorIterator(_shape, _container, _pos + n);
    }

    TENSOR_FUNC TensorIterator operator-(difference_type n) const
    {
      return TensorIterator(_shape, _container, _pos - n);
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

  private:
    shape_type _shape;
    mutable container_type _container;
    difference_type _pos;
  };

  template <typename Shape, typename Container>
  class BaseTensor
  {
  public:
    using value_type = typename Container::value_type;
    using reference = typename Container::reference;
    using const_reference = typename Container::const_reference;

    using shape_type = Shape;
    using container_type = Container;

    using iterator = TensorIterator<shape_type, typename container_traits<Container>::view_type>;
    using const_iterator = TensorIterator<shape_type, typename container_traits<Container>::const_view_type>;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    BaseTensor() = default;
    ~BaseTensor() = default;

    explicit TENSOR_FUNC BaseTensor(Shape shape_, Container container_) : _shape(shape_), _container(container_) {}

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

    static constexpr bool is_contiguous()
    {
      return Shape::is_contiguous() && container_traits<container_type>::is_contiguous;
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
    TENSOR_FUNC decltype(auto) operator[](index_t index)
    {
      return _container[_shape[index]];
    }

    /// @brief linear index access.
    TENSOR_FUNC decltype(auto) operator[](index_t index) const
    {
      return _container[_shape[index]];
    }

    /// @brief returns iterator to start of tensor.
    TENSOR_FUNC iterator begin()
    {
      using ct = container_traits<container_type>;
      return iterator(_shape, ct::make_view(_container), 0);
    }

    /// @brief returns iterator to start of tensor.
    TENSOR_FUNC const_iterator begin() const
    {
      using ct = container_traits<container_type>;
      return const_iterator(_shape, ct::make_view(_container), 0);
    }

    /// @brief returns iterator to the element following the last element of tensor.
    TENSOR_FUNC iterator end()
    {
      using ct = container_traits<container_type>;
      return iterator(_shape, ct::make_view(_container), size());
    }

    /// @brief returns iterator to the element following the last element of tensor.
    TENSOR_FUNC const_iterator end() const
    {
      using ct = container_traits<container_type>;
      return const_iterator(_shape, ct::make_view(_container), size());
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
    Container _container;

    TENSOR_FUNC decltype(auto) subview(index_t index)
    {
      return _container[index];
    }

    TENSOR_FUNC decltype(auto) subview(index_t index) const
    {
      return _container[index];
    }

    template <size_t N>
    TENSOR_FUNC auto subview(const std::array<span, N> &spans)
    {
      return make_subview(_container, spans);
    }

    template <size_t N>
    TENSOR_FUNC auto subview(const std::array<span, N> &spans) const
    {
      return make_subview(_container, spans);
    }

    TENSOR_FUNC auto subview(const span &s)
    {
      return make_subview(_container, s);
    }

    TENSOR_FUNC auto subview(const span &s) const
    {
      return make_subview(_container, s);
    }
  };

} // namespace tensor::details

#endif
