#pragma once
#include "TensorView/Containers/ContainerTraits.hpp"
#include "TensorView/Macros.hpp"
#include "TensorView/Utility/Memory.hpp"

namespace tensor::details
{
   template <typename T, MemorySpace MemSpace>
   class OwningContainer;

   template <typename T, MemorySpace MemSpace>
   class ViewContainer
   {
   public:
      using value_type = T;
      using reference = T &;
      using const_reference = const T &;
      using pointer = T *;

      static constexpr MemorySpace memory_space = MemSpace;

   private:
      pointer _ptr;
      index_t _capacity;

   public:
      constexpr ViewContainer() = default;
      constexpr ViewContainer(const ViewContainer &) = default;
      constexpr ViewContainer &operator=(const ViewContainer &) = default;
      constexpr ViewContainer(ViewContainer &&) = default;
      constexpr ViewContainer &operator=(ViewContainer &&) = default;
      constexpr ~ViewContainer() = default;

      template <typename U>
      constexpr ViewContainer(U *ptr, index_t capacity) : _ptr{static_cast<pointer>(ptr)}, _capacity{capacity}
      {
      }

      template <typename U, MemorySpace MS>
         requires(compatibleMemorySpaces(MemSpace, MS))
      constexpr ViewContainer(const ViewContainer<U, MS> &other)
          : _ptr{static_cast<pointer>(other.data())}, _capacity{other.capacity()}
      {
      }

      constexpr index_t capacity() const
      {
         return _capacity;
      }

      constexpr pointer data() const
      {
         return _ptr;
      }

      TENSOR_FUNC reference operator[](index_t index)
      {
         TENSOR_DEBUG_VERIFY_MEMORY_SPACE(MemSpace);

         TENSOR_DEBUG_ASSERT(_ptr != nullptr, printf("TensorView Error: attempting to dereference nullptr.\n"));

         TENSOR_DEBUG_ASSERT(index < _capacity, {
            printf("TensorView Error: index %ju out of bounds for ViewContainer of capacity %ju.\n",
                   static_cast<uintmax_t>(index), static_cast<uintmax_t>(_capacity));
         });

         return _ptr[index];
      }

      TENSOR_FUNC const_reference operator[](index_t index) const
      {
         TENSOR_DEBUG_VERIFY_MEMORY_SPACE(MemSpace);

         TENSOR_DEBUG_ASSERT(_ptr != nullptr, printf("TensorView Error: attempting to dereference nullptr.\n"));

         TENSOR_DEBUG_ASSERT(index < _capacity, {
            printf("TensorView Error: index %ju out of bounds for ViewContainer of capacity %ju.\n",
                   static_cast<uintmax_t>(index), static_cast<uintmax_t>(_capacity));
         });

         return _ptr[index];
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
   struct ContainerTraits<ViewContainer<T, MemSpace>>
   {
      using value_type = T;
      using container_type = ViewContainer<T, MemSpace>;
      using rview_type = ViewContainer<T, MemSpace>;
      using rcview_type = ViewContainer<const T, MemSpace>;

      static constexpr bool owning()
      {
         return false;
      }

      static constexpr bool mutableElements()
      {
         return std::assignable_from<T &, T>;
      }

      static constexpr MemorySpace memorySpace()
      {
         return MemSpace;
      }

      static TENSOR_FUNC rview_type makeRView(container_type &x)
      {
         return rview_type(x.data(), x.capacity());
      }

      static TENSOR_FUNC rcview_type makeRCView(const container_type &x)
      {
         return rcview_type(x.data(), x.capacity());
      }

      template <typename U, MemorySpace MS>
      static constexpr container_type from(const ViewContainer<U, MS> &other)
      {
         return container_type(other);
      }

      template <typename U, MemorySpace MS>
      static constexpr container_type from(const OwningContainer<U, MS> &other)
         requires(compatibleMemorySpaces(MemSpace, MS))
      {
         return container_type(other.data(), other.capacity());
      }
   };
} // namespace tensor::details