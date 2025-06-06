#ifndef __TENSOR_VIEW_TENSOR_TRAITS_HPP__
#define __TENSOR_VIEW_TENSOR_TRAITS_HPP__

#include "tensorview_config.hpp"
#include "FixedTensorShape.hpp"
#include "DynamicTensorShape.hpp"

namespace tensor::details
{
  template <typename T>
  struct tensor_traits;

  // compatibility of std::vector with tensor_traits
  template <typename scalar, typename Alloc>
  struct tensor_traits<std::vector<scalar, Alloc>>
  {
    using tensor_type = std::vector<scalar, Alloc>;
    using value_type = scalar;
    using shape_type = DynamicTensorShape<1>;
    using container_type = std::vector<scalar, Alloc>;

    static constexpr bool is_contiguous = true;
    static constexpr bool is_mutable = std::assignable_from<value_type &, value_type>;

    static shape_type shape(const container_type &container)
    {
      return shape_type(container.size());
    }

    static decltype(auto) container(const container_type &container)
    {
      return container;
    }

    static decltype(auto) container(container_type &container)
    {
      return container;
    }

    static decltype(auto) container(container_type &&container)
    {
      return std::move(container);
    }
  };

  template <typename scalar, size_t N>
  struct tensor_traits<std::array<scalar, N>>
  {
    using tensor_type = std::array<scalar, N>;
    using value_type = scalar;
    using shape_type = FixedTensorShape<N>;
    using container_type = std::array<scalar, N>;

    static constexpr bool is_contiguous = true;
    static constexpr bool is_mutable = std::assignable_from<value_type &, value_type>;

    static shape_type shape(const container_type &)
    {
      return {};
    }

    static decltype(auto) container(const container_type &container)
    {
      return container;
    }

    static decltype(auto) container(container_type &container)
    {
      return container;
    }

    static decltype(auto) container(container_type &&container)
    {
      return std::move(container);
    }
  };
} // namespace tensor::details

#endif
