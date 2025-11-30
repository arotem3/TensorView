#pragma once
#include "TensorView/Containers/ContainerTraits.hpp"
#include "TensorView/Containers/ViewContainer.hpp"
#include "TensorView/Macros.hpp"
#include "TensorView/Utility/Memory.hpp"

namespace tensor::details
{
   template <typename T, MemorySpace MemSpace = MemorySpace::Host>
   class OwningContainer
   {
   private:
      std::shared_ptr<T[]> _data;
      index_t _capacity;

      friend struct ContainerTraits<OwningContainer<T, MemSpace>>;

      template <typename U, MemorySpace MS>
      friend class OwningContainer;

   public:
      using value_type = T;
      using reference = T &;
      using const_reference = const T &;
      using pointer = T *;

      static constexpr MemorySpace memory_space = MemSpace;

      /**
       * @brief Constructs an OwningContainer with the specified capacity.
       */
      inline OwningContainer(index_t capacity = 0)
          : _data(tensor::allocate<T, MemSpace>(capacity), tensor::deleter<T, MemSpace>{}), _capacity{capacity}
      {
      }

      ~OwningContainer() = default;
      OwningContainer(const OwningContainer &) = default;
      OwningContainer &operator=(const OwningContainer &) = default;
      OwningContainer(OwningContainer &&) = default;
      OwningContainer &operator=(OwningContainer &&) = default;

      /**
       * @brief Constructs an OwningContainer sharing the data of another OwningContainer.
       */
      template <typename U, MemorySpace MS>
         requires(compatibleMemorySpaces(MemSpace, MS))
      inline OwningContainer(const OwningContainer<U, MS> &other) : _data(other._data), _capacity{other.capacity()}
      {
      }

      /**
       * @brief Returns a pointer to the underlying data.
       */
      inline pointer data() const
      {
         return _data.get();
      }

      /**
       * @brief Returns true if this container has a unique reference to its data or if the container is empty.
       */
      inline bool unique() const
      {
         return _data.use_count() <= 1;
      }

      /**
       * @brief Returns the capacity of the container.
       */
      constexpr index_t capacity() const
      {
         return _capacity;
      }

      /**
       * @brief Resizes the container to at least new_capacity.
       * If the new capacity is less than or equal to the current capacity, no action is taken.
       * Data is preserved.
       * Cannot resize if there are multiple references to the container.
       */
      inline void resize(index_t new_capacity)
      {
         TENSOR_CHECK(unique(), printf("Cannot resize OwningContainer with multiple references.\n"));

         if (new_capacity <= _capacity)
            return;

         auto old = _data;

         _data.reset(tensor::allocate<T, MemSpace>(new_capacity), tensor::deleter<T, MemSpace>{});

         if (old)
            tensor::details::copy_n(old.get(), _capacity, _data.get());

         _capacity = new_capacity;
      }

      /**
       * @brief Returns a reference to the element at the specified index.
       */
      inline reference operator[](index_t index)
      {
         TENSOR_DEBUG_VERIFY_MEMORY_SPACE(MemSpace);

         TENSOR_DEBUG_ASSERT(_data, printf("Attempting to dereference nullptr.\n"));

         TENSOR_DEBUG_ASSERT(index < _capacity,
                             printf("Index %ju out of bounds for OwningContainer of capacity %ju.\n",
                                    static_cast<uintmax_t>(index), static_cast<uintmax_t>(_capacity)));
         return _data[index];
      }

      /**
       * @brief Returns a const reference to the element at the specified index.
       */
      inline const_reference operator[](index_t index) const
      {
         TENSOR_DEBUG_VERIFY_MEMORY_SPACE(MemSpace);

         TENSOR_DEBUG_ASSERT(_data, printf("Attempting to dereference nullptr.\n"));

         TENSOR_DEBUG_ASSERT(index < _capacity,
                             printf("Index %ju out of bounds for OwningContainer of capacity %ju.\n",
                                    static_cast<uintmax_t>(index), static_cast<uintmax_t>(_capacity)));
         return _data[index];
      }

      /**
       * @brief Synchronizes the container's data to the specified memory space.
       * Namely, managed memory is prefetched to host or device.
       * If the container's memory space is not Managed, and to != memory_space, an error is raised.
       */
      inline void syncTo(MemorySpace to) const
      {
         tensor::synchronizeMemory(data(), capacity(), memory_space, to);
      }
   };

   template <typename T, MemorySpace MemSpace>
   struct ContainerTraits<OwningContainer<T, MemSpace>>
   {
      using value_type = T;
      using container_type = OwningContainer<T, MemSpace>;
      using rview_type = ViewContainer<T, MemSpace>;
      using rcview_type = ViewContainer<const T, MemSpace>;
      using pview_type = OwningContainer<T, MemSpace>;
      using pcview_type = OwningContainer<const T, MemSpace>;

      static constexpr bool owning()
      {
         return true;
      }

      static constexpr bool mutableElements()
      {
         return std::assignable_from<T &, T>;
      }

      static constexpr MemorySpace memorySpace()
      {
         return MemSpace;
      }

      static inline rview_type makeRView(container_type &x)
      {
         return rview_type(x.data(), x.capacity());
      }

      static inline rcview_type makeRCView(const container_type &x)
      {
         return rcview_type(x.data(), x.capacity());
      }

      static inline pview_type makePView(container_type &x)
      {
         return pview_type(x);
      }

      static inline pcview_type makePCView(const container_type &x)
      {
         return pcview_type(x);
      }

      template <typename U, MemorySpace MS>
      static container_type from(const OwningContainer<U, MS> &other)
      {
         return container_type(other);
      }
   };
} // namespace tensor::details