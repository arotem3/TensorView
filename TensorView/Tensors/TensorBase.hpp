#pragma once
#include "TensorView/Access/AccessPattern.hpp"
#include "TensorView/Access/Iterator.hpp"
#include "TensorView/Access/makePatternFrom.hpp"
#include "TensorView/Containers/ContainerTraits.hpp"
#include "TensorView/Containers/StaticContainer.hpp"
#include "TensorView/Tensors/TensorTraits.hpp"
#include "TensorView/Utility/Copy.hpp"
#include "TensorView/Utility/InitializerTensor.hpp"

namespace tensor::details
{
   template <typename AccessPattern, typename Container, bool Owner>
   class TensorBase;

   template <bool Owner, typename AccessPattern, typename Container>
   auto makeTensorBase(AccessPattern &&pattern, Container &&container)
   {
      using access_pattern_type = std::remove_cvref_t<AccessPattern>;
      using container_type = std::remove_cvref_t<Container>;
      return TensorBase<access_pattern_type, container_type, Owner>(std::forward<AccessPattern>(pattern),
                                                                    std::forward<Container>(container));
   }

   template <typename AccessPattern, typename Container>
   decltype(auto) makeView(AccessPattern &&pattern, Container &&container)
   {
      if constexpr (IndexLike<std::remove_cvref_t<AccessPattern>>)
         return container[pattern];
      else
      {
         using ct = ContainerTraits<std::remove_cvref_t<Container>>;
         return makeTensorBase<false>(std::forward<AccessPattern>(pattern),
                                      ct::makeView(std::forward<Container>(container)));
      }
   }

   template <typename AccessPattern, typename Container>
   decltype(auto) makeRView(AccessPattern &&pattern, Container &&container)
   {
      if constexpr (IndexLike<std::remove_cvref_t<AccessPattern>>)
         return container[pattern];
      else
      {
         using ct = ContainerTraits<std::remove_cvref_t<Container>>;
         return makeTensorBase<false>(std::forward<AccessPattern>(pattern),
                                      ct::makeRView(std::forward<Container>(container)));
      }
   }

   template <typename AccessPattern, typename Container, bool Owner>
   class TensorBase
   {
   public:
      using container_type = Container;
      using shape_type = AccessPattern;
      using tensor_type = TensorBase<AccessPattern, Container, Owner>;

      using value_type = typename container_type::value_type;

   protected:
      shape_type _access_pattern;
      container_type _container;

   private:
      template <typename T>
      friend struct TensorTraits;

      using ct = ContainerTraits<container_type>;

      static constexpr bool _is_mutable = ct::mutableElements();
      static constexpr bool _is_contiguous = is_contiguous_access_pattern<shape_type>;
      static constexpr index_t _num_dims = shape_type::numDims();
      static constexpr bool _is_raw = is_raw_container<container_type>; // raw containers have looser rules

      static_assert(!Owner || _is_mutable, "Owning TensorBase must have mutable elements to own data.");
      static_assert(!Owner || !_is_raw, "Owning TensorBase cannot have a raw container.");

   public:
      TensorBase() = default;
      ~TensorBase() = default;

      explicit TensorBase(shape_type &&shape_, container_type &&container_)
          : _access_pattern(std::move(shape_)), _container(std::move(container_))
      {
         TENSOR_CHECK(shape_.extent() <= container_.capacity(),
                      printf("Container capacity %ju smaller than shape extent %ju.\n",
                             static_cast<uintmax_t>(container_.capacity()), static_cast<uintmax_t>(shape_.extent())));
      }

      /**
       * @brief returns the number of dimensions of the tensor.
       */
      static constexpr index_t numDims()
      {
         return shape_type::numDims();
      }

      /**
       * @brief returns whether this TensorBase owns its data.
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
       * @brief the total number of elements in the tensor.
       */
      constexpr index_t size() const
      {
         return _access_pattern.size();
      }

      /**
       * @brief returns the size of the specified dimension.
       */
      constexpr index_t shape(index_t dim) const
      {
         return _access_pattern.shape(dim);
      }

      /**
       * @brief returns the shape object of the tensor.
       */
      constexpr const shape_type &shape() const
      {
         return _access_pattern;
      }

      /**
       * @brief returns whether the tensor is (logically) empty.
       */
      constexpr bool empty() const
      {
         return _access_pattern.size() == 0;
      }

      /**
       * @brief If Owner, performs a deep copy of other's data.
       * If not Owner, binds this view to other.
       */
      TensorBase(const TensorBase &other) : _access_pattern(other._access_pattern)
      {
         if constexpr (Owner)
         {
            _container = container_type(_access_pattern.extent());
            details::copyTensorToTensor(other, *this);
         }
         else
         {
            _container = other._container;
         }
      }

      /**
       * @brief If Owner, performs a deep copy of other's data.
       * If not Owner, binds this view to other.
       */
      TensorBase(TensorBase &&other) : _access_pattern(std::move(other._access_pattern))
      {
         if constexpr (Owner)
         {
            if (other.unique())
            {
               _container = std::move(other._container);
            }
            else
            {
               _container = container_type(_access_pattern.extent());
               details::copyTensorToTensor(other, *this);
            }
         }
         else
         {
            _container = std::move(other._container);
         }
      }

      /**
       * @brief If Owner, performs a deep copy of other's data.
       * If not Owner, binds this view to other.
       */
      template <typename OtherContainer, bool OtherOwner>
      TensorBase(TensorBase<shape_type, OtherContainer, OtherOwner> &&other)
         requires(std::is_convertible_v<OtherContainer, container_type>)
          : _access_pattern(std::move(TensorTraits<TensorBase<shape_type, OtherContainer, OtherOwner>>::shape(other)))
      {
         using other_traits = TensorTraits<TensorBase<shape_type, OtherContainer, OtherOwner>>;
         if constexpr (Owner)
         {
            if (other.unique())
            {
               _container = std::move(other_traits::container(other));
            }
            else
            {
               _container = container_type(_access_pattern.extent());
               details::copyTensorToTensor(other, *this);
            }
         }
         else
         {
            _container = ct::from(std::move(other_traits::container(other)));
         }
      }

      /**
       * @brief Deep copy
       */
      template <typename TensorType>
      TensorBase(TensorType &&other)
         requires(Owner && !IndexLike<std::remove_cvref_t<TensorType>>)
          : _access_pattern(makePatternLike<shape_type>(other.shape()))
      {
         if constexpr (!is_static_container<container_type>)
            _container = container_type(_access_pattern.extent());
         details::copyTensorToTensor(other, *this);
      }

      /**
       * @brief Binds this view to other.
       */
      template <typename TensorType>
      TensorBase(TensorType &&other)
         requires(!Owner && !IndexLike<std::remove_cvref_t<TensorType>>)
          : _access_pattern(makePatternFrom<shape_type>(other.shape()))
      {
         using traits = details::TensorTraits<std::remove_cvref_t<TensorType>>;
         _container = ct::from(traits::container(other));
      }

      /**
       * @brief Constructs a Tensor from nested initializer lists.
       *
       * @example 1D: 3-vector
       * Tensor<float, 1> t = {1.0f, 2.0f, 3.0f};
       *
       * @example 2D: 2x3 matrix
       * Tensor<float, 2> t = {{1.0f, 2.0f, 3.0f},{4.0f, 5.0f, 6.0f}};
       */
      TensorBase(typename InitializerTensor<value_type, _num_dims>::ListType &&list)
         requires(Owner)
      {
         InitializerTensor<value_type, _num_dims> init{std::move(list)};

         _access_pattern = details::makePatternLike<shape_type>(init);
         _container = container_type(_access_pattern.extent());

         details::fromInitializer<value_type, _num_dims>(raw(), std::move(init));
      }

      /**
       * @brief Constructs an owning Tensor with the specified shape dimensions.
       *
       * @example Tensor<float, 2> t(3, 4); // creates a 3x4 tensor of floats
       */
      template <IndexLike... Dimensions>
      explicit TensorBase(Dimensions... dims)
         requires(Owner)
          : _access_pattern(makePattern<shape_type>(dims...)), _container(_access_pattern.extent())
      {}

      /**
       * @brief Constructs a TensorBase that views the specified data pointer with the specified shape dimensions.
       * Applicable only for owning Tensors or Raw Views.
       * If Owner, the Tensor takes ownership of the data pointer. If the underlying pointer is deleted elsewhere,
       * behavior is undefined.
       * If not Owner (Raw View), the TensorBase simply views the data pointer. The user must ensure that the data
       * pointer remains valid for the lifetime of the TensorBase.
       */
      template <typename U, IndexLike... Dimensions>
      explicit TensorBase(U *data_ptr, Dimensions... dims)
         requires(std::is_convertible_v<U *, value_type *> && is_standard_pattern<shape_type> && (Owner || _is_raw))
          : _access_pattern(makePattern<shape_type>(dims...)), _container(data_ptr, _access_pattern.extent())
      {}

      template <typename U>
      explicit TensorBase(U *data_ptr)
         requires(std::is_convertible_v<U *, value_type *> && is_static_pattern<shape_type> && _is_raw)
          : _access_pattern(), _container(data_ptr, _access_pattern.extent())
      {}

      /**
       * @brief Elementwise copy. DOES NOT REBIND VIEWS.
       * If Owner, then reshapes and copies data. Ownership of the old data is released. Active persistent/reference
       * views remain valid; raw views become undefined behavior. If not Owner, then copies data if shapes match,
       * otherwise raises an error.
       */
      TensorBase &operator=(const TensorBase &other)
         requires(_is_mutable)
      {
         if constexpr (Owner && !is_static_pattern<shape_type>)
         {
            _access_pattern = other._access_pattern;
            _container = container_type(_access_pattern.extent());
         }

         details::copyTensorToTensor(other, *this);

         return *this;
      }

      /**
       * @brief Elementwise copy. DOES NOT REBIND VIEWS.
       * If Owner, then moves if unique, otherwise reshapes and copies data. Ownership of the old data is released.
       * Active persistent/reference views remain valid; raw views become undefined behavior. If not Owner, then copies
       * data if shapes match, otherwise raises an error.
       */
      TensorBase &operator=(TensorBase &&other)
         requires(_is_mutable)
      {
         if constexpr (Owner)
         {
            _access_pattern = std::move(other._access_pattern);

            if (other.unique())
            {
               _container = std::move(other._container);
               return *this;
            }
            else
            {
               _container.resize(_access_pattern.extent());
            }
         }

         details::copyTensorToTensor(other, *this);
         return *this;
      }

      /**
       * @brief Elementwise copy from any tensor-like object. DOES NOT REBIND VIEWS.
       * If Owner, then reshapes and copies data. Ownership of the old data is released. Active persistent/reference
       * views remain valid; raw views become undefined behavior. If not Owner, then copies data if shapes match,
       * otherwise raises an error.
       */
      template <typename TensorType>
      TensorBase &operator=(const TensorType &other)
         requires(_is_mutable)
      {
         if constexpr (Owner && !is_static_pattern<shape_type>)
         {
            _access_pattern = makePatternLike<shape_type>(other.shape());
            _container.resize(_access_pattern.extent());
         }
         details::copyTensorToTensor(other, *this);
         return *this;
      }

      /**
       * @brief Assignment from nested initializer lists.
       * If Owner, then reshapes and copies data. Ownership of the old data is released. Active persistent/reference
       * views remain valid; raw views become undefined behavior. If not Owner, then copies data if shapes match,
       * otherwise raises an error.
       *
       * @example 1D: 3-vector
       * Tensor<float, 1> t;
       * t = {1.0f, 2.0f, 3.0f};
       *
       * @example 2D: 2x3 matrix
       * Tensor<float, 2> t;
       * t = {{1.0f, 2.0f, 3.0f},{4.0f, 5.0f, 6.0f}};
       */
      TensorBase &operator=(typename InitializerTensor<value_type, _num_dims>::ListType &&list)
         requires(_is_mutable)
      {
         InitializerTensor<value_type, _num_dims> init{std::move(list)};

         if constexpr (Owner && !is_static_pattern<shape_type>)
         {
            _access_pattern = details::makePatternLike<shape_type>(init);
            _container.resize(_access_pattern.extent());
         }

         details::fromInitializer<value_type, _num_dims>(raw(), std::move(init));
         return *this;
      }

      /**
       * @brief If not Owner, rebinds this view to other.
       */
      template <typename TensorType>
      TensorBase &rebind(TensorType &&other)
         requires(!Owner)
      {
         using traits = details::TensorTraits<std::remove_cvref_t<TensorType>>;
         _access_pattern = makePatternFrom<shape_type>(traits::shape(other));
         _container = ct::from(traits::container(other));

         return *this;
      }

      /**
       * @brief Claims ownership of another TensorBase's data if that data is uniquely owned.
       * Otherwise, raises an error. Ownership of the old data is released. Active persistent/reference
       * views remain valid; raw views become undefined behavior.
       */
      template <typename OtherContainer, bool OtherOwner>
      TensorBase &claim(TensorBase<shape_type, OtherContainer, OtherOwner> &&other)
         requires(Owner && std::is_convertible_v<OtherContainer, container_type>)
      {
         TENSOR_CHECK(other.unique(), printf("Cannot claim ownership of TensorBase with multiple references.\n"));

         _access_pattern = std::move(other._access_pattern);
         _container = std::move(other._container);

         return *this;
      }

      /**
       * @brief Changes the shape of the tensor preserving the data.
       * If the new shape requires fewer elements, no allocation is performed.
       * If the new shape requires more elements, the container is reallocated and the data is copied.
       * Cannot resize if there are active persistent/reference views to the data.
       * If there are active raw views to the data, behavior is undefined for those views.
       */
      template <IndexLike... Dimensions>
      TensorBase &reshape(Dimensions... new_access_pattern)
         requires(Owner && !is_static_pattern<shape_type>)
      {
         TENSOR_CHECK(_container.unique(), printf("Cannot reshape TensorBase with multiple references.\n"));
         _access_pattern = makePattern<shape_type>(new_access_pattern...);
         _container.resize(_access_pattern.size());
         return *this;
      }

      /**
       * @brief Synchronizes the container's data to the specified memory space.
       * Namely, managed memory is prefetched to host or device.
       * If the container's memory space is not Managed, and to != memory_space, an error is raised.
       */
      void syncTo(MemorySpace to) const
      {
         _container.syncTo(to);
      }

      /**
       * @brief returns a raw view of the data with the same shape.
       */
      auto raw()
      {
         return makeRView(shape_type(_access_pattern), _container);
      }

      /**
       * @brief returns a const raw view of the data with the same shape.
       */
      auto raw() const
      {
         return makeRView(shape_type(_access_pattern), _container);
      }

      /**
       * @brief returns a persistent/reference view of the data with the same shape.
       */
      auto view()
      {
         return makeView(shape_type(_access_pattern), _container);
      }

      /**
       * @brief returns a const persistent/reference view of the data with the same shape.
       */
      auto view() const
      {
         return makeView(shape_type(_access_pattern), _container);
      }

      /**
       * @brief returns a pointer to the first element of the underlying data.
       */
      value_type *data()
         requires(_is_contiguous)
      {
         return _container.data() + _access_pattern.offset();
      }

      /**
       * @brief returns a const pointer to the first element of the underlying data.
       */
      const value_type *data() const
         requires(_is_contiguous)
      {
         return _container.data() + _access_pattern.offset();
      }

      /**
       * @brief implicit conversion to a raw pointer. Same as data().
       */
      operator value_type *()
         requires(_is_contiguous)
      {
         return data();
      }

      /**
       * @brief implicit conversion to a const raw pointer. Same as data().
       */
      operator const value_type *() const
         requires(_is_contiguous)
      {
         return data();
      }

      /**
       * @brief element access with multi-dimensional indices.
       * For fancy index expressions, returns a persistent/reference view if the underlying container is persistent,
       * otherwise returns a raw view.
       */
      template <typename... Indices>
         requires(sizeof...(Indices) == _num_dims &&
                  !(sizeof...(Indices) == 1 &&
                    (std::is_same_v<std::remove_cvref_t<Indices>, std::array<index_t, _num_dims>> || ...)))
      decltype(auto) at(Indices &&...indices)
      {
         return makeView(_access_pattern.at(std::forward<Indices>(indices)...), ct::makeView(_container));
      }

      /**
       * @brief element access with multi-dimensional indices.
       * For fancy index expressions, returns a const persistent/reference view if the underlying container is
       * persistent, otherwise returns a const raw view.
       */
      template <typename... Indices>
         requires(sizeof...(Indices) == _num_dims &&
                  !(sizeof...(Indices) == 1 &&
                    (std::is_same_v<std::remove_cvref_t<Indices>, std::array<index_t, _num_dims>> || ...)))
      decltype(auto) at(Indices &&...indices) const
      {
         return makeView(_access_pattern.at(std::forward<Indices>(indices)...), ct::makeView(_container));
      }

      /**
       * @brief element access with multi-dimensional indices passed as a single std::array.
       */
      decltype(auto) at(const std::array<index_t, _num_dims> &multi_index)
      {
         return std::apply([&](const auto &...indices) -> decltype(auto) { return at(indices...); }, multi_index);
      }

      /**
       * @brief element access with multi-dimensional indices passed as a single std::array.
       */
      decltype(auto) at(const std::array<index_t, _num_dims> &multi_index) const
      {
         return std::apply([&](const auto &...indices) -> decltype(auto) { return at(indices...); }, multi_index);
      }

      /**
       * @brief element access operators with multi-dimensional indices.
       * For fancy index expressions, returns a persistent/reference view if the underlying container is persistent,
       * otherwise returns a raw view.
       */
      template <typename... Indices>
         requires(sizeof...(Indices) == _num_dims)
      decltype(auto) operator()(Indices &&...indices)
      {
         return at(std::forward<Indices>(indices)...);
      }

      /**
       * @brief element access operators with multi-dimensional indices.
       * For fancy index expressions, returns a const persistent/reference view if the underlying container is
       * persistent, otherwise returns a const raw view.
       */
      template <typename... Indices>
         requires(sizeof...(Indices) == _num_dims)
      decltype(auto) operator()(Indices &&...indices) const
      {
         return at(std::forward<Indices>(indices)...);
      }

      /**
       * @brief element access operators with linear indices.
       */
      decltype(auto) operator[](index_t index)
      {
         return _container[_access_pattern[index]];
      }

      /**
       * @brief element access operators with linear indices.
       */
      decltype(auto) operator[](index_t index) const
      {
         return _container[_access_pattern[index]];
      }

      /**
       * @brief returns an iterator to the beginning of the tensor data.
       */
      auto begin()
      {
         if constexpr (_is_contiguous)
            return data();
         else
            return TensorBegin(*this);
      }

      /**
       * @brief returns a const iterator to the beginning of the tensor data.
       */
      auto begin() const
      {
         if constexpr (_is_contiguous)
            return data();
         else
            return TensorBegin(*this);
      }

      /**
       * @brief returns an iterator to the end of the tensor data.
       */
      auto end()
      {
         if constexpr (_is_contiguous)
            return data() + size();
         else
            return TensorEndSentinel{};
         // return TensorEnd(*this);
      }

      /**
       * @brief returns a const iterator to the end of the tensor data.
       */
      auto end() const
      {
         if constexpr (_is_contiguous)
            return data() + size();
         else
            return TensorEndSentinel{};
         // return TensorEnd(*this);
      }

      /**
       * @brief returns a reverse iterator to the beginning of the reversed tensor data.
       */
      auto rbegin()
      {
         return std::make_reverse_iterator(end());
      }

      /**
       * @brief returns a const reverse iterator to the beginning of the reversed tensor data.
       */
      auto rbegin() const
      {
         return std::make_reverse_iterator(end());
      }

      /**
       * @brief returns a reverse iterator to the end of the reversed tensor data.
       */
      auto rend()
      {
         return std::make_reverse_iterator(begin());
      }

      /**
       * @brief returns a const reverse iterator to the end of the reversed tensor data.
       */
      auto rend() const
      {
         return std::make_reverse_iterator(begin());
      }
   };

   template <typename AccessPattern, typename Container, bool Owner>
   struct TensorTraits<TensorBase<AccessPattern, Container, Owner>> : std::true_type
   {
      using tensor_type = TensorBase<AccessPattern, Container, Owner>;
      using container_type = Container;
      using shape_type = AccessPattern;

      using value_type = typename container_type::value_type;

      using container_traits = ContainerTraits<container_type>;

      static constexpr bool contiguous()
      {
         return is_contiguous_access_pattern<shape_type>;
      }

      static constexpr index_t numDims()
      {
         return shape_type::numDims();
      }

      static constexpr bool mutableElements()
      {
         return container_traits::mutableElements();
      }

      static constexpr const shape_type &shape(const tensor_type &tensor)
      {
         return tensor._access_pattern;
      }

      static constexpr shape_type &shape(tensor_type &tensor)
      {
         return tensor._access_pattern;
      }

      static constexpr shape_type shape(tensor_type &&tensor)
      {
         return std::move(tensor._access_pattern);
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
