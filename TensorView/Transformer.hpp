#ifndef __TENSOR_VIEW_TRANSFORMER_HPP__
#define __TENSOR_VIEW_TRANSFORMER_HPP__

#include "tensorview_config.hpp"
#include "BaseTensor.hpp"
#include "ContainerTraits.hpp"
#include "TensorTraits.hpp"

namespace tensor::details
{
  template <typename Inner, typename Outer>
  struct _ComposedFunction
  {
    Inner _inner;
    Outer _outer;
    
    TENSOR_HOST_DEVICE inline decltype(auto) operator()(const auto &value) const
    {
      return _outer(_inner(value));
    }
  };

  template <typename Container, typename Func>
  struct TransformedContainerWrapper
  {
  public:
    using container_type = std::remove_cvref_t<Container>;
    using function_type = Func;
    using original_value_type = std::remove_cvref_t<typename container_type::value_type>;
    using value_type = std::invoke_result_t<const Func, original_value_type>;

    container_type _container;
    function_type _func;

    TENSOR_HOST_DEVICE inline TransformedContainerWrapper(const TransformedContainerWrapper &) = default;
    TENSOR_HOST_DEVICE inline TransformedContainerWrapper &operator=(const TransformedContainerWrapper &) = default;
    TENSOR_HOST_DEVICE inline TransformedContainerWrapper(TransformedContainerWrapper &&) = default;
    TENSOR_HOST_DEVICE inline TransformedContainerWrapper &operator=(TransformedContainerWrapper &&) = default;

    TENSOR_HOST_DEVICE inline TransformedContainerWrapper(Container &&container, Func &&func)
        : _container(std::forward<Container>(container)), _func(std::forward<Func>(func)) {}

    TENSOR_HOST_DEVICE inline value_type operator[](index_t index) const
    {
      original_value_type value = _container[index];
      return _func(value);
    }
  };

  template <typename Container, typename Func>
  struct container_traits<TransformedContainerWrapper<Container, Func>>
  {
  private:
    using ct = container_traits<Container>;

  public:
    static constexpr bool is_contiguous = false;
    static constexpr bool is_mutable = false;

    using value_type = typename TransformedContainerWrapper<Container, Func>::value_type;
    using view_type = TransformedContainerWrapper<typename ct::view_type, Func>;
    using const_view_type = TransformedContainerWrapper<typename ct::const_view_type, Func>;

    TENSOR_HOST_DEVICE static inline view_type make_view(TransformedContainerWrapper<Container, Func> &x, ptrdiff_t offset = 0)
    {
      return view_type(ct::make_view(x._container, offset), Func{x._func});
    }

    TENSOR_HOST_DEVICE static inline const_view_type make_view(const TransformedContainerWrapper<Container, Func> &x, ptrdiff_t offset = 0)
    {
      return const_view_type(ct::make_view(x._container, offset), Func{x._func});
    }
  };
} // namespace tensor::details

namespace tensor
{
  template <typename Func, typename Shape, typename Container>
  class Transformer : public details::BaseTensor<Shape, details::TransformedContainerWrapper<Container, Func>>
  {
  public:
    using shape_type = Shape;
    using container_type = details::TransformedContainerWrapper<Container, Func>;
    using base_tensor = details::BaseTensor<shape_type, container_type>;

    using value_type = typename container_type::value_type;

    static constexpr bool is_contiguous()
    {
      return false;
    }

    TENSOR_HOST_DEVICE inline Transformer(Shape &&shape, Container &&container, Func &&func)
        : base_tensor(std::forward<Shape>(shape), container_type(std::forward<Container>(container), std::forward<Func>(func))) {}

    Transformer(const Transformer &) = default;
    Transformer &operator=(const Transformer &) = default;
    Transformer(Transformer &&) = default;
    Transformer &operator=(Transformer &&) = default;

  private:
    friend struct details::tensor_traits<Transformer<Func, Shape, Container>>;
  };

  /**
   * @brief Creates a view object which when indexed applies the given function to each element of the tensor.
   */
  template <typename Func, typename TensorType>
  TENSOR_HOST_DEVICE inline auto transform(Func &&func, TensorType &&tensor)
  {
    using tensor_t = std::remove_cvref_t<TensorType>;
    using shape_type = typename details::tensor_traits<tensor_t>::shape_type;
    using container_type = typename details::tensor_traits<tensor_t>::container_type;

    if constexpr (std::is_lvalue_reference_v<TensorType &&>)
    {
      using ct = details::container_traits<container_type>;
      using view_type = typename ct::const_view_type;
      const auto& container = details::tensor_traits<tensor_t>::container(tensor);
      return Transformer<Func, shape_type, view_type>(
          details::tensor_traits<tensor_t>::shape(tensor),
          ct::make_view(container),
          std::forward<Func>(func));
    }
    else
    {
      return Transformer<Func, shape_type, container_type>(
          details::tensor_traits<tensor_t>::shape(tensor),
          details::tensor_traits<tensor_t>::container(std::forward<TensorType>(tensor)),
          std::forward<Func>(func));
    }
  }

  template <typename G, typename F, typename Shape, typename Container>
  TENSOR_HOST_DEVICE inline auto transform(G &&g, Transformer<F, Shape, Container> &&tensor)
  {
    using tensor_t = std::remove_cvref_t<Transformer<F, Shape, Container>>;
    auto container = details::tensor_traits<tensor_t>::container(std::forward<tensor_t>(tensor));

    using value_type = typename details::container_traits<Container>::value_type;

    auto gf = [g = std::forward<G>(g), f = std::move(container._func)](value_type x)
    {
      return g(f(x));
    };

    return Transformer<decltype(gf), Shape, Container>(
        details::tensor_traits<tensor_t>::shape(tensor),
        std::move(container._container),
        std::move(gf));
  }

  template <typename G, typename F, typename Shape, typename Container>
  TENSOR_HOST_DEVICE inline auto transform(G &&g, const Transformer<F, Shape, Container> &tensor)
  {
    using tensor_t = std::remove_cvref_t<Transformer<F, Shape, Container>>;
    const auto &container = details::tensor_traits<tensor_t>::container(tensor);
    using view_type = typename details::container_traits<Container>::const_view_type;

    using composed_t = details::_ComposedFunction<F, G>;

    return Transformer<composed_t, Shape, view_type>(
        details::tensor_traits<tensor_t>::shape(tensor),
        details::container_traits<Container>::make_view(container._container),
        composed_t{container._func, std::forward<G>(g)});
  }

  template <typename G, typename F, typename Shape, typename Container>
  TENSOR_HOST_DEVICE inline auto transform(G &&g, Transformer<F, Shape, Container> &tensor)
  {
    return transform(std::forward<G>(g), static_cast<const Transformer<F, Shape, Container> &>(tensor));
  }
} // namespace tensor

namespace tensor::details
{
  template <typename Func, typename Shape, typename Container>
  struct tensor_traits<tensor::Transformer<Func, Shape, Container>>
  {
    using tensor_type = tensor::Transformer<Func, Shape, Container>;
    using value_type = typename tensor_type::value_type;
    using shape_type = typename tensor_type::shape_type;
    using container_type = typename tensor_type::container_type;

    static constexpr bool is_contiguous = false;

    static shape_type shape(const tensor_type &tensor)
    {
      return tensor._shape;
    }

    static const container_type &container(const tensor_type &tensor)
    {
      return tensor._container;
    }

    static container_type &container(tensor_type &tensor)
    {
      return tensor._container;
    }

    static container_type container(tensor_type &&tensor)
    {
      return std::move(tensor._container);
    }
  };
} // namespace tensor::details

#endif
