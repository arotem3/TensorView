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
  template <size_t Rank, typename Container>
  class SubView : public details::BaseTensor<details::StridedShape<Rank>, Container>
  {
  public:
    using base_tensor = details::BaseTensor<details::StridedShape<Rank>, Container>;
    using shape_type = details::StridedShape<Rank>;
    using container_type = Container;

    TENSOR_FUNC explicit SubView(shape_type &&shape, Container &&view) : base_tensor(std::move(shape), std::move(view)) {}
  };

  template <typename T, size_t Rank>
  using SimpleSubView = SubView<Rank, details::ViewContainer<T>>;
} // namespace tensor

#endif
