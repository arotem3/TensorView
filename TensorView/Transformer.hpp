#ifndef __TENSOR_VIEW_TRANSFORMER_HPP__
#define __TENSOR_VIEW_TRANSFORMER_HPP__

#include "tensorview_config.hpp"
#include "BaseTensor.hpp"
#include "ContainerTraits.hpp"

namespace tensor::details
{
    template <typename Container, typename Func>
    class TransformedContainerWrapper
    {
    public:
        using container_type = std::remove_cvref_t<Container>;
        using function_type = Func;
        using original_value_type = std::remove_cvref_t<typename container_type::value_type>;
        using value_type = std::invoke_result_t<const Func, original_value_type>;

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

    private:
        Container _container;
        function_type _func;

        friend struct container_traits<TransformedContainerWrapper<Container, Func>>;
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
            return view_type(ct::make_view(x._container, offset), x._func);
        }

        TENSOR_HOST_DEVICE static inline const const_view_type make_view(const TransformedContainerWrapper<Container, Func> &x, ptrdiff_t offset = 0)
        {
            return const_view_type(ct::make_view(x._container, offset), x._func);
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

        static constexpr bool is_contiguous()
        {
            return false;
        }

        TENSOR_HOST_DEVICE inline Transformer(Shape &&shape, Container &&container, Func &&func)
            : base_tensor(std::forward<Shape>(shape), container_type(std::forward<Container>(container), std::forward<Func>(func))) {}

        Transformer(const Transformer&) = default;
        Transformer &operator=(const Transformer &) = default;
        Transformer(Transformer &&) = default;
        Transformer &operator=(Transformer &&) = default;
    };

    /**
     * @brief Creates a view object which when indexed applies the given function to each element of the tensor.
     */
    template <typename Func, typename TensorType>
    TENSOR_HOST_DEVICE inline auto transform(Func &&func, TensorType &&tensor)
    {
        using tensor_t = std::remove_cvref_t<TensorType>;
        using shape_type = typename tensor_t::shape_type;
        using container_type = typename tensor_t::container_type;
        using ct = details::container_traits<container_type>;
        using view_type = typename ct::const_view_type;

        view_type view = ct::make_view(static_cast<const container_type &>(tensor.container()));

        return Transformer<Func, shape_type, view_type>(
            std::forward<shape_type>(tensor.shape()),
            std::move(view),
            std::forward<Func>(func));
    }
} // namespace tensor


#endif
