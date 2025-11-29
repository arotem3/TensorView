#pragma once
#include "TensorView/Access/Iterator.hpp"
#include "TensorView/Access/MakeView.hpp"
#include "TensorView/Containers/ContainerTraits.hpp"
#include "TensorView/Containers/OwningContainer.hpp"
#include "TensorView/Macros.hpp"
#include "TensorView/Shapes/ShapeTraits.hpp"
#include "TensorView/Tensors/TensorTraits.hpp"
#include "TensorView/Utility/Copy.hpp"

namespace tensor::details
{
   template <typename Shape, typename T, MemorySpace MemSpace, bool Owner>
   class PersistentView
   {
   public:
      using container_type = OwningContainer<T, MemSpace>;
      using shape_type = Shape;
      using tensor_type = PersistentView<Shape, T, MemSpace, Owner>;

      friend struct TensorTraits<tensor_type>;

   private:
      using ct = ContainerTraits<container_type>;
      using st = ShapeTraits<shape_type>;

      static constexpr bool _is_mutable = ct::mutableElements();
      static constexpr bool _is_contiguous = st::contiguous();

      static_assert(_is_mutable || !Owner, "Immutable PersistentView cannot own data.");

   public:
      using value_type = typename ct::value_type;

      using iterator = TensorIterator<shape_type, typename ct::rview_type>;
      using const_iterator = TensorIterator<shape_type, typename ct::rcview_type>;
      using reverse_iterator = std::reverse_iterator<iterator>;
      using const_reverse_iterator = std::reverse_iterator<const_iterator>;

      using rview = RawView<shape_type, T, MemSpace>;
      using const_rview = RawView<shape_type, std::add_const_t<T>, MemSpace>;
      using pview = PersistentView<shape_type, T, MemSpace, false>;
      using const_pview = PersistentView<shape_type, std::add_const_t<T>, MemSpace, false>;

   protected:
      shape_type _shape;
      container_type _container;

   public:
      PersistentView() = default;
      ~PersistentView() = default;

      explicit PersistentView(shape_type &&shape_, container_type &&container_)
          : _shape(std::move(shape_)), _container(std::move(container_))
      {
         TENSOR_CHECK(shape_.size() <= container_.capacity(),
                      printf("Container capacity %ju smaller than shape size %ju.\n",
                             static_cast<uintmax_t>(container_.capacity()), static_cast<uintmax_t>(shape_.size())));
      }

      /**
       * @brief If Owner, performs a deep copy of other's data.
       * If not Owner, performs a shallow copy of other's shape and container.
       */
      PersistentView(const PersistentView &other)
      {
         if constexpr (Owner)
         {
            _shape = other._shape;
            _container = container_type(other.size());
            details::copyTensorToTensor(other, *this);
         }
         else
         {
            _shape = other._shape;
            _container = other._container;
         }
      }

      /**
       * @brief If Owner, moves other's data if unique, otherwise performs a deep copy.
       * If not Owner, performs a shallow move of other's shape and container.
       */
      PersistentView(PersistentView &&other)
      {
         if constexpr (Owner)
         {
            _shape = std::move(other._shape);

            if (other.unique())
            {
               _container = std::exchange(other._container, container_type());
            }
            else
            {
               _container = container_type(other.size());
               details::copyTensorToTensor(other, *this);
            }
         }
         else
         {
            _shape = std::move(other._shape);
            _container = std::exchange(other._container, container_type());
         }
      }

      /**
       * @brief If Owner, performs a deep copy of other's data.
       * If not Owner, performs a shallow copy of other's shape and container.
       */
      template <typename S1, typename T1, MemorySpace MS1, bool Owner1>
      PersistentView(PersistentView<S1, T1, MS1, Owner1> &&other)
      {
         if constexpr (Owner)
         {
            _shape = st::from(other.shape());

            if (other.unique())
            {
               using other_container_type = typename PersistentView<S1, T1, MS1, Owner1>::container_type;
               _container = std::exchange(other._container, other_container_type());
            }
            else
            {
               _container = container_type(other.size());
               details::copyTensorToTensor(other, *this);
            }
         }
         else
         {
            rebind(std::forward<PersistentView<S1, T1, MS1, Owner1>>(other));
         }
      }

      /**
       * @brief If Owner, performs a deep copy of other's data.
       * If not Owner, performs a shallow copy of other's shape and container.
       */
      template <typename TensorType>
      PersistentView(TensorType &&other)
         requires(TensorTraits<std::remove_cvref_t<TensorType>>::value)
      {
         if constexpr (Owner)
         {
            using traits = TensorTraits<std::remove_cvref_t<TensorType>>;
            _shape = st::from(traits::shape(other));
            _container = container_type(_shape.size());
            details::copyTensorToTensor(other, *this);
         }
         else
         {
            rebind(std::forward<TensorType>(other));
         }
      }

      /**
       * @brief Constructs a PersistentView with the specified shape.
       */
      template <IndexLike... Sizes>
      explicit PersistentView(Sizes... shape_)
         requires(Owner)
          : _shape(shape_...), _container(_shape.size())
      {
      }

      /**
       * @brief Constructs a PersistentView claiming ownership of a raw pointer's data with the specified shape.
       * If the underlying pointer is deleted elsewhere, behavior is undefined.
       */
      template <IndexLike... Sizes>
      explicit PersistentView(const T *data, Sizes... shape_)
         requires(Owner)
          : _shape(shape_...), _container(data, _shape.size())
      {
         TENSOR_CHECK(data != nullptr, printf("Cannot claim ownership of nullptr data.\n"));
      }

      /**
       * @brief Copy assignment operator.
       * If Owner, then reshapes and copies data. Ownership of the old data is released. Active persistent/reference
       * views remain valid; raw views become undefined behavior. If not Owner, then copies data if shapes match,
       * otherwise raises an error.
       */
      PersistentView &operator=(const PersistentView &other)
         requires(_is_mutable)
      {
         if constexpr (Owner)
         {
            _shape = other._shape;
            _container = container_type(other.size());
         }

         details::copyTensorToTensor(other, *this);

         return *this;
      }

      /**
       * @brief Move assignment operator.
       * If Owner, then moves if unique, otherwise reshapes and copies data. Ownership of the old data is released.
       * Active persistent/reference views remain valid; raw views become undefined behavior. If not Owner, then copies
       * data if shapes match, otherwise raises an error.
       */
      PersistentView &operator=(PersistentView &&other)
      {
         if constexpr (Owner)
         {
            _shape = other._shape;

            if (other.unique())
            {
               _container = std::exchange(other._container, container_type());
            }
            else
            {
               _container = container_type(other.size());
               details::copyTensorToTensor(other, *this);
            }
         }
         else
         {
            static_assert(_is_mutable, "Cannot copy to an immutable PersistentView.");
            details::copyTensorToTensor(other, *this);
         }

         return *this;
      }

      /**
       * @brief Copy assignment from any tensor-like object.
       * If Owner, then reshapes and copies data. Ownership of the old data is released. Active persistent/reference
       * views remain valid; raw views become undefined behavior. If not Owner, then copies data if shapes match,
       * otherwise raises an error.
       */
      template <typename TensorType>
      PersistentView &operator=(const TensorType &other)
         requires(_is_mutable)
      {
         if constexpr (Owner)
         {
            using traits = TensorTraits<std::remove_cvref_t<TensorType>>;
            _shape = st::from(traits::shape(other));
            _container = container_type(_shape.size());
         }

         details::copyTensorToTensor(other, *this);

         return *this;
      }

      /**
       * @brief rebinds the view to another tensor-like object's data.
       * Can only bind other PersistentView types.
       */
      template <typename TensorType>
      PersistentView &rebind(TensorType &&other)
         requires(!Owner)
      {
         using traits = TensorTraits<std::remove_cvref_t<TensorType>>;
         this->_shape = st::from(traits::shape(other));
         this->_container = ct::from(traits::container(other));
         return *this;
      }

      /**
       * @brief Claims ownership of another PersistentView's data if that data is uniquely owned.
       * Otherwise, raises an error. Ownership of the old data is released. Active persistent/reference
       * views remain valid; raw views become undefined behavior.
       */
      template <typename S1, typename T1, MemorySpace MS1, bool Owner1>
      PersistentView &claim(const PersistentView<S1, T1, MS1, Owner1> &other)
         requires(Owner)
      {
         TENSOR_CHECK(other.unique(), printf("Cannot claim ownership of PersistentView with multiple references.\n"));

         _shape = st::from(other.shape());

         using other_container_type = typename PersistentView<S1, T1, MS1, Owner1>::container_type;
         _container = std::exchange(other._container, other_container_type());

         return *this;
      }

      /**
       * @brief Claims ownership of a RawView's data.
       * If the underlying pointer is deleted elsewhere, behavior is undefined.
       * Ownership of the old data is released. Active persistent/reference views remain valid; raw views become
       * undefined behavior.
       */
      template <typename S1, typename T1, MemorySpace MS1>
      PersistentView &claim(const RawView<S1, T1, MS1> &other)
         requires(Owner)
      {
         _shape = st::from(other.shape());
         _container = container_type(other.data(), other.size());

         return *this;
      }

      /**
       * @brief Claims ownership of a raw pointer's data with the specified shape.
       * If the underlying pointer is deleted elsewhere, behavior is undefined.
       * Ownership of the old data is released. Active persistent/reference views remain valid; raw views become
       * undefined behavior.
       */
      template <typename T1, IndexLike... Sizes>
      PersistentView &claim(T1 *data, Sizes... shape)
         requires(Owner)
      {
         TENSOR_CHECK(data != nullptr, printf("Cannot claim ownership of nullptr data.\n"));
         _shape = shape_type(shape...);
         _container = container_type(data, _shape.size());

         return *this;
      }

      /**
       * @brief returns the number of dimensions.
       */
      static constexpr index_t numDims()
      {
         return shape_type::numDims();
      }

      /**
       * @brief returns whether this PersistentView owns its data.
       */
      static constexpr bool owning()
      {
         return Owner;
      }

      /**
       * @brief returns whether the container's data is uniquely owned.
       */
      bool unique() const
      {
         return _container.unique();
      }

      /**
       * @brief returns whether the data is contiguous in the specified linear order.
       */
      bool contiguous(LinearOrder O) const
      {
         return _shape.contiguous(O);
      }

      /**
       * @brief returns the total number of elements.
       */
      index_t size() const
      {
         return _shape.size();
      }

      /**
       * @brief returns the size of the specified dimension.
       */
      index_t shape(index_t dim) const
      {
         return _shape.shape(dim);
      }

      /**
       * @brief returns the shape object.
       */
      const shape_type &shape() const
      {
         return _shape;
      }

      /**
       * @brief returns whether the tensor is empty.
       */
      bool empty() const
      {
         return _shape.empty();
      }

      /**
       * @brief Changes the shape of the tensor preserving the data.
       * If the new shape requires fewer elements, no allocation is performed.
       * If the new shape requires more elements, the container is reallocated and the data is copied.
       * Cannot resize if there are active persistent/reference views to the data.
       * If there are active raw views to the data, behavior is undefined for those views.
       */
      template <IndexLike... Sizes>
      PersistentView &reshape(Sizes... new_shape)
         requires(Owner)
      {
         TENSOR_CHECK(_container.unique(), printf("Cannot reshape PersistentView with multiple references.\n"));
         _shape = shape_type(new_shape...);
         _container.resize(_shape.size());

         return *this;
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
       * @brief returns a raw view of the data with the same shape.
       */
      rview raw()
      {
         return rview(*this);
      }

      /**
       * @brief returns a const raw view of the data with the same shape.
       */
      const_rview raw() const
      {
         return const_rview(*this);
      }

      /**
       * @brief returns a persistent/reference view of the data with the same shape.
       * This view extends the lifetime of the data as long as the PersistentView exists.
       */
      pview view()
      {
         return pview(*this);
      }

      /**
       * @brief returns a const persistent/reference view of the data with the same shape.
       * This view extends the lifetime of the data as long as the PersistentView exists.
       */
      const_pview view() const
      {
         return const_pview(*this);
      }

      /**
       * @brief returns a pointer to the data.
       */
      value_type *data()
         requires(_is_contiguous)
      {
         return _container.data();
      }

      /**
       * @brief returns a const pointer to the data.
       */
      const value_type *data() const
         requires(_is_contiguous)
      {
         return _container.data();
      }

      /**
       * @brief implicit conversion to a pointer to the data.
       */
      operator value_type *()
         requires(_is_contiguous)
      {
         return data();
      }

      /**
       * @brief implicit conversion to a const pointer to the data.
       */
      operator const value_type *() const
         requires(_is_contiguous)
      {
         return data();
      }

      /**
       * @brief element access with multi-dimensional indices. Returns a persistent/reference view for fancy indexing.
       */
      template <typename... Indices>
      decltype(auto) at(Indices... indices) TENSOR_REQUIRES_NOT_DEVICE_SPACE(MemSpace)
      {
         return makeSubView(*this, _shape(std::forward<Indices>(indices)...));
      }

      /**
       * @brief element access with multi-dimensional indices. Returns a const persistent/reference view for fancy
       * indexing.
       */
      template <typename... Indices>
      decltype(auto) at(Indices... indices) const TENSOR_REQUIRES_NOT_DEVICE_SPACE(MemSpace)
      {
         return makeSubView(*this, _shape(std::forward<Indices>(indices)...));
      }

      /**
       * @brief element access operators with multi-dimensional indices. Returns a persistent/reference view for fancy
       * indexing.
       */
      template <typename... Indices>
      decltype(auto) operator()(Indices... indices) TENSOR_REQUIRES_NOT_DEVICE_SPACE(MemSpace)
      {
         return at(std::forward<Indices>(indices)...);
      }

      /**
       * @brief element access operators with multi-dimensional indices. Returns a const persistent/reference view for
       * fancy indexing.
       */
      template <typename... Indices>
      decltype(auto) operator()(Indices... indices) const TENSOR_REQUIRES_NOT_DEVICE_SPACE(MemSpace)
      {
         return at(std::forward<Indices>(indices)...);
      }

      /**
       * @brief element access operators with linear indices.
       */
      decltype(auto) operator[](index_t index) TENSOR_REQUIRES_NOT_DEVICE_SPACE(MemSpace)
      {
         return _container[_shape[index]];
      }

      /**
       * @brief element access operators with linear indices.
       */
      decltype(auto) operator[](index_t index) const TENSOR_REQUIRES_NOT_DEVICE_SPACE(MemSpace)
      {
         return _container[_shape[index]];
      }

      /**
       * @brief returns an iterator to the beginning.
       */
      iterator begin()
      {
         return iterator(_shape, ct::makeRView(_container), 0);
      }

      /**
       * @brief returns a const iterator to the beginning.
       */
      const_iterator begin() const
      {
         return const_iterator(_shape, ct::makeRCView(_container), 0);
      }

      /**
       * @brief returns an iterator to the end.
       */
      iterator end()
      {
         return iterator(_shape, ct::makeRView(_container), size());
      }

      /**
       * @brief returns a const iterator to the end.
       */
      const_iterator end() const
      {
         return const_iterator(_shape, ct::makeRCView(_container), size());
      }

      /**
       * @brief returns a reverse iterator to the beginning.
       */
      reverse_iterator rbegin()
      {
         return reverse_iterator(end());
      }

      /**
       * @brief returns a const reverse iterator to the beginning.
       */
      const_reverse_iterator rbegin() const
      {
         return const_reverse_iterator(end());
      }

      /**
       * @brief returns a reverse iterator to the end.
       */
      reverse_iterator rend()
      {
         return reverse_iterator(begin());
      }

      /**
       * @brief returns a const reverse iterator to the end.
       */
      const_reverse_iterator rend() const
      {
         return const_reverse_iterator(begin());
      }
   };

   template <typename Shape, typename T, MemorySpace MemSpace, bool Owner>
   struct TensorTraits<PersistentView<Shape, T, MemSpace, Owner>> : std::true_type
   {
      using tensor_type = PersistentView<Shape, T, MemSpace, Owner>;
      using container_type = typename tensor_type::container_type;
      using shape_type = typename tensor_type::shape_type;

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
