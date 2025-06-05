#ifndef __TENSOR_VIEW_CONTAINER_TRAITS_HPP__
#define __TENSOR_VIEW_CONTAINER_TRAITS_HPP__

#include "tensorview_config.hpp"
#include "ViewContainer.hpp"

namespace tensor::details
{
  /**
   * @brief Traits class template for container types.
   *
   * Specialize this struct for your container type to provide compile-time information
   * and utilities required by the TensorView framework.
   *
   * The specialization must define the following members:
   * - `static constexpr bool is_contiguous`: True if the container stores elements contiguously in memory.
   *
   * - `static constexpr bool is_mutable`: True if the container allows modification of its elements.
   *
   * - `typename value_type`: The type of elements stored in the container.
   *
   * - `typename view_type`: The type representing a mutable view of the container.
   *
   * - `typename const_view_type`: The type representing a read-only view of the container.
   *
   * - `static view_type make_view(T& container, ptrdiff_t offset = 0)`: Returns a mutable view of the given container starting at offset.
   *
   * - `static const_view_type make_view(const T& container, ptrdiff_t offset = 0)`: Returns a read-only view of the given container starting at offset.
   *
   * @tparam T The container type to specialize for.
   */
  template <typename T>
  struct container_traits;

  template <typename T, typename Alloc>
  struct container_traits<std::vector<T, Alloc>>
  {
    static constexpr bool is_contiguous = true;
    static constexpr bool is_mutable = std::assignable_from<T &, T>;
    using value_type = T;
    using view_type = ViewContainer<T>;
    using const_view_type = ViewContainer<const T>;

    TENSOR_HOST_DEVICE static inline view_type make_view(std::vector<T, Alloc> &x, ptrdiff_t offset = 0)
    {
      return view_type(x.data() + offset);
    }

    TENSOR_HOST_DEVICE static inline const_view_type make_view(const std::vector<T, Alloc> &x, ptrdiff_t offset = 0)
    {
      return const_view_type(x.data() + offset);
    }
  };

  template <typename T, size_t N>
  struct container_traits<std::array<T, N>>
  {
    static constexpr bool is_contiguous = true;
    static constexpr bool is_mutable = std::assignable_from<T &, T>;
    using value_type = T;
    using view_type = ViewContainer<T>;
    using const_view_type = ViewContainer<const T>;

    TENSOR_HOST_DEVICE static inline view_type make_view(std::array<T, N> &x, ptrdiff_t offset = 0)
    {
      return view_type(x.data() + offset);
    }

    TENSOR_HOST_DEVICE static inline const_view_type make_view(const std::array<T, N> &x, ptrdiff_t offset = 0)
    {
      return const_view_type(x.data() + offset);
    }
  };

  template <typename T>
  struct container_traits<ViewContainer<T>>
  {
    static constexpr bool is_contiguous = true;
    static constexpr bool is_mutable = std::assignable_from<T &, T>;
    using value_type = T;
    using view_type = ViewContainer<T>;
    using const_view_type = ViewContainer<const T>;

    TENSOR_HOST_DEVICE static inline view_type make_view(ViewContainer<T> &x, ptrdiff_t offset = 0)
    {
      return view_type(x.data() + offset);
    }

    TENSOR_HOST_DEVICE static inline const_view_type make_view(const ViewContainer<T> &x, ptrdiff_t offset = 0)
    {
      return const_view_type(x.data() + offset);
    }
  };
} // namespace tensor::details

#endif
