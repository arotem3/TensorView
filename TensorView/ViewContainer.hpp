#ifndef __TENSOR_VIEW_VIEW_CONTAINER_HPP__
#define __TENSOR_VIEW_VIEW_CONTAINER_HPP__

#include "tensorview_config.hpp"
#include "errors.hpp"
#include "span.hpp"

namespace tensor::details
{
  template <typename T>
  class ViewContainer
  {
  public:
    using value_type = T;
    using reference = T &;
    using const_reference = TENSOR_CONST_QUAL(T) &;
    using pointer = T *;
    using const_pointer = TENSOR_CONST_QUAL(T) *;

    TENSOR_FUNC ViewContainer(pointer data = nullptr) : ptr{data} {}

    TENSOR_FUNC const_pointer data() const
    {
      return ptr;
    }

    TENSOR_FUNC pointer data()
    {
      return ptr;
    }

    TENSOR_FUNC reference operator[](index_t index)
    {
#ifdef TENSOR_DEBUG
      if (ptr == nullptr)
        tensor_bad_memory_access();
#endif
      return ptr[index];
    }

    TENSOR_FUNC const_reference operator[](index_t index) const
    {
#ifdef TENSOR_DEBUG
      if (ptr == nullptr)
        tensor_bad_memory_access();
#endif
      return ptr[index];
    }

  private:
    pointer ptr;
  };

  template <typename Container>
  auto wrap_view(Container &container)
  {
    using value_type = typename Container::value_type;
    return details::ViewContainer<value_type>(container.data());
  }

  template <typename Container>
  auto wrap_view(const Container &container)
  {
    return details::ViewContainer<const typename Container::value_type>(container.data());
  }
} // namespace tensor::details

#endif
