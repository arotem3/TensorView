#pragma once
#include "TensorView/Containers/StaticContainer.hpp"
#include "TensorView/Shapes/StaticShape.hpp"
#include "TensorView/Tensors/RawView.hpp"

namespace tensor
{
   /**
    * @brief A tensor with static shape and static storage.
    */
   template <typename T, LinearOrder Order, index_t... Dims>
   class FCStaticTensor;

   template <typename T, index_t... Dims>
   using FStaticTensor = FCStaticTensor<T, LinearOrder::F, Dims...>;

   template <typename T, index_t... Dims>
   using CStaticTensor = FCStaticTensor<T, LinearOrder::C, Dims...>;

   template <typename T, index_t... Dims>
   using StaticTensor = FStaticTensor<T, Dims...>;

   template <typename T, LinearOrder Order, index_t... Dims>
   class FCStaticTensor
   {
   public:
      using shape_type = details::StaticShape<Order, Dims...>;
      using container_type = details::StaticContainer<T, shape_type::extent()>;
      using tensor_type = FCStaticTensor<T, Order, Dims...>;

   private:
      using ct = details::ContainerTraits<container_type>;
      using st = details::ShapeTraits<shape_type>;
      using traits = details::TensorTraits<tensor_type>;

      friend struct details::TensorTraits<tensor_type>;

      static constexpr bool _is_mutable = ct::mutableElements();
      static constexpr index_t _num_dims = shape_type::numDims();

   public:
      using value_type = typename traits::value_type;

      using iterator = details::TensorIterator<shape_type, typename ct::rview_type>;
      using const_iterator = details::TensorIterator<shape_type, typename ct::rcview_type>;
      using reverse_iterator = std::reverse_iterator<iterator>;
      using const_reverse_iterator = std::reverse_iterator<const_iterator>;

      using rview = details::RawView<shape_type, T, MemorySpace::Unspecified>;
      using const_rview = details::RawView<shape_type, std::add_const_t<T>, MemorySpace::Unspecified>;

   protected:
      shape_type _shape;
      container_type _container;

   public:
      FCStaticTensor() = default;
      ~FCStaticTensor() = default;

      constexpr FCStaticTensor(shape_type, container_type &&container_) : _shape(), _container(std::move(container_))
      {
         TENSOR_CHECK(_shape.size() <= _container.capacity(),
                      printf("FCStaticTensor size %ju exceeds container capacity %ju.\n",
                             static_cast<uintmax_t>(_shape.size()), static_cast<uintmax_t>(_container.capacity())));
      }

      constexpr FCStaticTensor(const FCStaticTensor &other) = default;
      constexpr FCStaticTensor(FCStaticTensor &&other) noexcept = default;

      constexpr FCStaticTensor(const details::InitializerTensor<T, _num_dims>::ListType &init)
      {
         details::fromInitializer<T, _num_dims>(raw(), init);
      }

      /**
       * @brief Element-wise copy.
       */
      constexpr FCStaticTensor &operator=(const FCStaticTensor &other)
         requires(_is_mutable)
      {
         _container = other._container;
         return *this;
      }

      /**
       * @brief Element-wise copy.
       */
      constexpr FCStaticTensor &operator=(FCStaticTensor &&other) noexcept
         requires(_is_mutable)
      {
         _container = std::move(other._container);
         return *this;
      }

      /**
       * @brief Elementwise copy from any tensor-like object
       */
      template <typename TensorType>
      constexpr FCStaticTensor &operator=(const TensorType &other)
         requires(_is_mutable)
      {
         for (index_t i = 0; i < _shape.size(); ++i)
         {
            _container[i] = static_cast<value_type>(other[i]);
         }

         return *this;
      }

      /**
       * @brief Elementwise copy from initializer list
       */
      constexpr FCStaticTensor &operator=(const typename details::InitializerTensor<T, _num_dims>::ListType &init)
         requires(_is_mutable)
      {
         details::fromInitializer<T, _num_dims>(raw(), init);
         return *this;
      }

      /**
       * @brief returns the number of dimensions of the tensor.
       */
      static constexpr index_t numDims()
      {
         return shape_type::numDims();
      }

      /**
       * @brief Is the tensor F-contiguous?
       */
      constexpr bool contiguous() const
      {
         return _shape.contiguous();
      }

      /**
       * @brief returns the total number of elements in the tensor.
       */
      constexpr index_t size() const
      {
         return _shape.size();
      }

      /**
       * @brief returns the size of the specified dimension.
       */
      constexpr index_t shape(index_t dim) const
      {
         return _shape.shape(dim);
      }

      /**
       * @brief returns the shape of the tensor.
       */
      constexpr shape_type shape() const
      {
         return _shape;
      }

      /**
       * @brief Is the tensor logically empty (i.e., has zero elements)?
       */
      constexpr bool empty() const
      {
         return _shape.empty();
      }

      /**
       * @brief creates a raw view of the same data with the same shape.
       */
      constexpr rview raw()
      {
         return rview(shape_type(), ct::makeRView(_container));
      }

      /**
       * @brief creates a const raw view of the same data with the same shape.
       */
      constexpr const_rview raw() const
      {
         return const_rview(shape_type(), ct::makeRCView(_container));
      }

      /**
       * @brief returns a pointer to the underlying data.
       */
      constexpr value_type *data()
      {
         return _container.data() + _shape.offset();
      }

      /**
       * @brief returns a pointer to the underlying data.
       */
      constexpr const value_type *data() const
      {
         return _container.data() + _shape.offset();
      }

      /**
       * @brief implicit conversion to a raw pointer.
       */
      constexpr operator value_type *()
      {
         return data();
      }

      /**
       * @brief implicit conversion to a const raw pointer.
       */
      constexpr operator const value_type *() const
      {
         return data();
      }

      /**
       * @brief element access with multi-dimensional indices.
       */
      template <typename... Indices>
      constexpr decltype(auto) at(Indices... indices)
      {
         return makeRawSubView(raw(), _shape(std::forward<Indices>(indices)...));
      }

      /**
       * @brief element access with multi-dimensional indices.
       */
      template <typename... Indices>
      constexpr decltype(auto) at(Indices... indices) const
      {
         return makeRawSubView(raw(), _shape(std::forward<Indices>(indices)...));
      }

      /**
       * @brief element access with multi-dimensional indices.
       */
      template <typename... Indices>
      constexpr decltype(auto) operator()(Indices... indices)
      {
         return at(std::forward<Indices>(indices)...);
      }

      /**
       * @brief element access with multi-dimensional indices.
       */
      template <typename... Indices>
      constexpr decltype(auto) operator()(Indices... indices) const
      {
         return at(std::forward<Indices>(indices)...);
      }

      /**
       * @brief element access with linear indices.
       */
      constexpr decltype(auto) operator[](index_t index)
      {
         return _container[_shape[index]];
      }

      /**
       * @brief element access with linear indices.
       */
      constexpr decltype(auto) operator[](index_t index) const
      {
         return _container[_shape[index]];
      }

      /**
       * @brief returns an iterator to the beginning.
       */
      constexpr iterator begin()
      {
         return iterator(_shape, ct::makeRView(_container), 0);
      }

      /**
       * @brief returns a const iterator to the beginning.
       */
      constexpr const_iterator begin() const
      {
         return const_iterator(_shape, ct::makeRCView(_container), 0);
      }

      /**
       * @brief returns an iterator to the end.
       */
      constexpr iterator end()
      {
         return iterator(_shape, ct::makeRView(_container), size());
      }

      /**
       * @brief returns a const iterator to the end.
       */
      constexpr const_iterator end() const
      {
         return const_iterator(_shape, ct::makeRCView(_container), size());
      }

      /**
       * @brief returns a reverse iterator to the beginning.
       */
      constexpr reverse_iterator rbegin()
      {
         return reverse_iterator(end());
      }

      /**
       * @brief returns a const reverse iterator to the beginning.
       */
      constexpr const_reverse_iterator rbegin() const
      {
         return const_reverse_iterator(end());
      }

      /**
       * @brief returns a reverse iterator to the end.
       */
      constexpr reverse_iterator rend()
      {
         return reverse_iterator(begin());
      }

      /**
       * @brief returns a const reverse iterator to the end.
       */
      constexpr const_reverse_iterator rend() const
      {
         return const_reverse_iterator(begin());
      }
   };
} // namespace tensor

namespace tensor::details
{
   template <typename T, LinearOrder Order, index_t... Dims>
   struct TensorTraits<FCStaticTensor<T, Order, Dims...>> : std::true_type
   {
      using tensor_type = FCStaticTensor<T, Order, Dims...>;
      using shape_type = details::StaticShape<Order, Dims...>;
      using container_type = details::StaticContainer<T, shape_type::extent()>;

      using value_type = T;

      using container_traits = ContainerTraits<container_type>;
      using shape_traits = ShapeTraits<shape_type>;

      static constexpr bool contiguous()
      {
         return shape_traits::contiguous();
      }

      static constexpr bool mutableElements()
      {
         return container_traits::mutableElements();
      }

      static constexpr shape_type shape(const tensor_type &)
      {
         return shape_type();
      }

      static constexpr const container_type &container(const tensor_type &tensor)
      {
         return tensor._container;
      }

      static constexpr container_type &container(tensor_type &tensor)
      {
         return tensor._container;
      }

      static constexpr container_type container(tensor_type &&tensor)
      {
         return std::move(tensor._container);
      }
   };
} // namespace tensor::details
