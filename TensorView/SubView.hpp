#ifndef __TENSOR_VIEW_SUB_VIEW_HPP__
#define __TENSOR_VIEW_SUB_VIEW_HPP__

#include "tensorview_config.hpp"
#include "span.hpp"
#include "ViewContainer.hpp"
#include "StridedShape.hpp"
#include "BaseTensor.hpp"
#include "TensorTraits.hpp"

namespace tensor
{
  /// @brief high dimensional sub-view into tensors
  /// @tparam scalar type of tensor elements
  /// @tparam Rank the dimension of the subview
  template <size_t Rank, typename Container>
  class SubView : public details::BaseTensor<details::StridedShape<Rank>, Container>
  {
  public:
    using base_tensor = details::BaseTensor<details::StridedShape<Rank>, Container>;
    using shape_type = details::StridedShape<Rank>;
    using container_type = Container;

    TENSOR_FUNC explicit SubView(shape_type &&shape, Container &&view) : base_tensor(std::move(shape), std::move(view)) {}

  private:
    friend struct details::tensor_traits<SubView<Rank, Container>>;
  };

  template <typename T, size_t Rank>
  using SimpleSubView = SubView<Rank, details::ViewContainer<T>>;
} // namespace tensor

namespace tensor::details
{
  template <size_t Rank, typename Container>
  struct tensor_traits<tensor::SubView<Rank, Container>>
  {
    using tensor_type = tensor::SubView<Rank, Container>;
    using value_type = typename tensor_type::container_type::value_type;
    using shape_type = typename tensor_type::shape_type;
    using container_type = typename tensor_type::container_type;

    static constexpr bool is_contiguous = false; // SubViews are not contiguous by default
    static constexpr bool is_mutable = std::assignable_from<value_type &, value_type>;

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
