#pragma once
#include "TensorView/Access/Iterator.hpp"
#include "TensorView/Access/MakeView.hpp"
#include "TensorView/Containers/ContainerTraits.hpp"
#include "TensorView/Containers/ViewContainer.hpp"
#include "TensorView/Macros.hpp"
#include "TensorView/Shapes/ShapeTraits.hpp"
#include "TensorView/Shapes/StaticShape.hpp"
#include "TensorView/Tensors/TensorTraits.hpp"
#include "TensorView/Utility/Copy.hpp"
#include "TensorView/Utility/InitializerTensor.hpp"

namespace tensor::details
{
   /**
    * @brief RawView represents a non-owning view of a multi-dimensional array of elements of type T in the specified
    * memory space.
    *
    * @tparam Shape The shape type describing the dimensions and layout of the tensor
    * @tparam T The type of elements in the tensor
    * @tparam MemSpace The memory space where the tensor data is stored
    */
   template <typename Shape, typename T, MemorySpace MemSpace>
   class RawView
   {
   public:
      using container_type = ViewContainer<T, MemSpace>;
      using shape_type = Shape;
      using tensor_type = RawView<Shape, T, MemSpace>;

   private:
      using ct = ContainerTraits<container_type>;
      using st = ShapeTraits<shape_type>;
      using traits = TensorTraits<tensor_type>;

      friend struct TensorTraits<tensor_type>;

      static constexpr bool _is_mutable = ct::mutableElements();
      static constexpr bool _is_contiguous = st::contiguous();
      static constexpr index_t _num_dims = shape_type::numDims();

   public:
      using value_type = typename traits::value_type;

      using iterator = TensorIterator<shape_type, typename ct::rview_type>;
      using const_iterator = TensorIterator<shape_type, typename ct::rcview_type>;
      using reverse_iterator = std::reverse_iterator<iterator>;
      using const_reverse_iterator = std::reverse_iterator<const_iterator>;

      using rview = tensor_type;
      using const_rview = RawView<shape_type, std::add_const_t<T>, MemSpace>;

   protected:
      shape_type _shape;
      container_type _container;

   public:
      RawView() = default;
      ~RawView() = default;

      TENSOR_FUNC explicit RawView(shape_type &&shape_, container_type &&container_)
          : _shape(std::move(shape_)), _container(std::move(container_))
      {
         TENSOR_CHECK(shape_.size() <= container_.capacity(),
                      printf("Container capacity %ju smaller than shape size %ju.\n",
                             static_cast<uintmax_t>(container_.capacity()), static_cast<uintmax_t>(shape_.size())));
      }

      TENSOR_FUNC RawView(const RawView &) = default;
      TENSOR_FUNC RawView(RawView &&) = default;

      /**
       * @brief bind to another tensor-like object's data.
       */
      template <typename TensorType>
      TENSOR_FUNC RawView(TensorType &&other)
         requires(TensorTraits<std::remove_cvref_t<TensorType>>::value)
      {
         rebind(std::forward<TensorType>(other));
      }

      /**
       * @brief construct from raw pointer and dimensions.
       * Specialized only for StandardShape. i.e. TensorView
       */
      template <typename U, IndexLike... Dims>
      TENSOR_FUNC explicit RawView(U *data_ptr, Dims... dims)
         requires(details::IsStandardShape<shape_type>::value)
          : _shape(dims...), _container(data_ptr, _shape.size())
      {
      }

      template <typename U>
      TENSOR_FUNC explicit RawView(U *data_ptr)
         requires(details::IsStaticShape<shape_type>::value)
          : _shape(), _container(data_ptr, _shape.size())
      {
      }

      /**
       * @brief Elementwise copy
       */
      RawView &operator=(const RawView &other)
         requires(_is_mutable)
      {
         details::copyTensorToTensor(other, *this);
         return *this;
      }

      /**
       * @brief Elementwise copy.
       * DOES NOT REBIND THE VIEW. Use rebind() to rebind the view to another memory region.
       */
      RawView &operator=(RawView &&other)
         requires(_is_mutable)
      {
         details::copyTensorToTensor(other, *this);
         return *this;
      }

      /**
       * @brief Elementwise copy from any tensor-like object
       */
      template <typename TensorType>
      RawView &operator=(const TensorType &other)
         requires(_is_mutable)
      {
         details::copyTensorToTensor(other, *this);
         return *this;
      }

      TENSOR_FUNC RawView &operator=(const typename details::InitializerTensor<T, _num_dims>::ListType &init)
         requires(_is_mutable)
      {
         fromInitializer<T, _num_dims>(*this, init);
         return *this;
      }

      /**
       * @brief rebinds the view to another tensor-like object's data.
       */
      template <typename TensorType>
      TENSOR_FUNC RawView &rebind(TensorType &&other)
      {
         using traits = details::TensorTraits<std::remove_cvref_t<TensorType>>;
         this->_shape = st::from(traits::shape(other));
         this->_container = ct::from(traits::container(other));
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
      TENSOR_FUNC bool contiguous() const
      {
         return _shape.contiguous();
      }

      /**
       * @brief returns the total number of elements in the tensor.
       */
      TENSOR_FUNC index_t size() const
      {
         return _shape.size();
      }

      /**
       * @brief returns the size of the specified dimension.
       */
      TENSOR_FUNC index_t shape(index_t dim) const
      {
         return _shape.shape(dim);
      }

      /**
       * @brief returns the shape object of the tensor.
       */
      TENSOR_FUNC const shape_type &shape() const
      {
         return _shape;
      }

      /**
       * @brief returns whether the tensor is empty.
       */
      TENSOR_FUNC bool empty() const
      {
         return _shape.empty();
      }

      /**
       * @brief Synchronizes the container's data to the specified memory space.
       * Namely, managed memory is prefetched to host or device.
       * If the container's memory space is not Managed, and ms != memory_space, an error is raised.
       */
      inline void syncTo(MemorySpace to) const
      {
         _container.syncTo(to);
      }

      /**
       * @brief creates a raw view of the same data with the same shape.
       */
      TENSOR_FUNC rview raw()
      {
         return tensor_type(*this);
      }

      /**
       * @brief creates a const raw view of the same data with the same shape.
       */
      TENSOR_FUNC const_rview raw() const
      {
         return const_rview(_shape, _container);
      }

      /**
       * @brief returns a pointer to the underlying data.
       */
      TENSOR_FUNC value_type *data()
         requires(_is_contiguous)
      {
         return _container.data() + _shape.offset();
      }

      /**
       * @brief returns a const pointer to the underlying data.
       */
      TENSOR_FUNC const value_type *data() const
         requires(_is_contiguous)
      {
         return _container.data() + _shape.offset();
      }

      /**
       * @brief implicit conversion to a raw pointer.
       */
      TENSOR_FUNC operator value_type *()
         requires(_is_contiguous)
      {
         return data();
      }

      /**
       * @brief implicit conversion to a const raw pointer.
       */
      TENSOR_FUNC operator const value_type *() const
         requires(_is_contiguous)
      {
         return data();
      }

      /**
       * @brief element access with multi-dimensional indices.
       */
      template <typename... Indices>
      TENSOR_FUNC decltype(auto) at(Indices... indices)
      {
         return makeRawSubView(*this, _shape(std::forward<Indices>(indices)...));
      }

      /**
       * @brief element access with multi-dimensional indices.
       */
      template <typename... Indices>
      TENSOR_FUNC decltype(auto) at(Indices... indices) const
      {
         return makeRawSubView(*this, _shape(std::forward<Indices>(indices)...));
      }

      /**
       * @brief element access operators with multi-dimensional indices.
       */
      template <typename... Indices>
      TENSOR_FUNC decltype(auto) operator()(Indices... indices)
      {
         return at(std::forward<Indices>(indices)...);
      }

      /**
       * @brief element access operators with multi-dimensional indices.
       */
      template <typename... Indices>
      TENSOR_FUNC decltype(auto) operator()(Indices... indices) const
      {
         return at(std::forward<Indices>(indices)...);
      }

      /**
       * @brief element access operators with linear indices.
       */
      TENSOR_FUNC decltype(auto) operator[](index_t index)
      {
         return _container[_shape[index]];
      }

      /**
       * @brief element access operators with linear indices.
       */
      TENSOR_FUNC decltype(auto) operator[](index_t index) const
      {
         return _container[_shape[index]];
      }

      /**
       * @brief returns an iterator to the beginning.
       */
      inline iterator begin()
      {
         return iterator(_shape, ct::makeRView(_container), 0);
      }

      /**
       * @brief returns a const iterator to the beginning.
       */
      inline const_iterator begin() const
      {
         return const_iterator(_shape, ct::makeRCView(_container), 0);
      }

      /**
       * @brief returns an iterator to the end.
       */
      inline iterator end()
      {
         return iterator(_shape, ct::makeRView(_container), size());
      }

      /**
       * @brief returns a const iterator to the end.
       */
      inline const_iterator end() const
      {
         return const_iterator(_shape, ct::makeRCView(_container), size());
      }

      /**
       * @brief returns a reverse iterator to the beginning.
       */
      inline reverse_iterator rbegin()
      {
         return reverse_iterator(end());
      }

      /**
       * @brief returns a const reverse iterator to the beginning.
       */
      inline const_reverse_iterator rbegin() const
      {
         return const_reverse_iterator(end());
      }

      /**
       * @brief returns a reverse iterator to the end.
       */
      inline reverse_iterator rend()
      {
         return reverse_iterator(begin());
      }

      /**
       * @brief returns a const reverse iterator to the end.
       */
      inline const_reverse_iterator rend() const
      {
         return const_reverse_iterator(begin());
      }
   };

   template <typename Shape, typename T, MemorySpace MemSpace>
   struct TensorTraits<RawView<Shape, T, MemSpace>> : std::true_type
   {
      using tensor_type = RawView<Shape, T, MemSpace>;
      using shape_type = Shape;
      using container_type = ViewContainer<T, MemSpace>;

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

      static constexpr shape_type shape(const tensor_type &tensor)
      {
         return tensor._shape;
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
