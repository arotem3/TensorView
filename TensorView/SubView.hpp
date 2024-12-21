#ifndef __TENSOR_VIEW_SUB_VIEW_HPP__
#define __TENSOR_VIEW_SUB_VIEW_HPP__

#include "tensorview_config.hpp"
#include "span.hpp"
#include "ViewContainer.hpp"
#include "StridedShape.hpp"
#include "BaseTensor.hpp"

namespace tensor
{
  /// @brief high dimensional sub-view into tensors
  /// @tparam scalar type of tensor elements
  /// @tparam Rank the dimension of the subview
  template <typename scalar, size_t Rank>
  class SubView : public details::BaseTensor<details::StridedShape<Rank>, details::ViewContainer<scalar>>
  {
  public:
    TENSOR_FUNC explicit SubView(details::ViewContainer<scalar> view, const std::array<span, Rank> &spans) : base_tensor(shape_type(spans), container_type(view.data() + details::offset(spans))) {}

  private:
    using base_tensor = details::BaseTensor<details::StridedShape<Rank>, details::ViewContainer<scalar>>;
    using shape_type = details::StridedShape<Rank>;
    using container_type = details::ViewContainer<scalar>;
  };

  template <typename scalar>
  class SubView<scalar, 1> : public details::BaseTensor<details::StridedShape<1>, details::ViewContainer<scalar>>
  {
  public:
    TENSOR_FUNC explicit SubView(details::ViewContainer<scalar> view, const span &s) : base_tensor(shape_type(s), container_type(view.data() + s.begin)) {}

  private:
    using base_tensor = details::BaseTensor<details::StridedShape<1>, details::ViewContainer<scalar>>;
    using shape_type = details::StridedShape<1>;
    using container_type = details::ViewContainer<scalar>;
  };
} // namespace tensor

#endif
