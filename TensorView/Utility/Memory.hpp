#pragma once
#include <algorithm>
#include <memory>
#include <stdexcept>

#include "TensorView/Macros.hpp"

namespace tensor
{
   enum class MemorySpace
   {
      Host, // Memory explicitly on the host
#ifdef TENSOR_USE_CUDA
      Device,  // Memory on a device (e.g., GPU) if available, otherwise on the host (for compatibility)
      Managed, // Unified memory accessible from both host and device
#endif
      Unspecified // Unspecified memory space. Cannot be used for allocations. Allows for unsafe access.
   };

   inline constexpr const char *memorySpaceToString(MemorySpace ms)
   {
      switch (ms)
      {
         case MemorySpace::Host:
            return "Host";
#ifdef TENSOR_USE_CUDA
         case MemorySpace::Device:
            return "Device";
         case MemorySpace::Managed:
            return "Managed";
#endif
         case MemorySpace::Unspecified:
            return "Unspecified";
         default:
            return "Unknown";
      }
   }

   /// @brief Allocates an array of a given size on the specified memory space.
   template <typename T, MemorySpace m = MemorySpace::Host>
   inline T *allocate(size_t n)
   {
      static_assert(m != MemorySpace::Unspecified, "Cannot allocate memory in Unspecified memory space.");

      if (n == 0)
         return nullptr;

      T *ptr = nullptr;

#ifdef TENSOR_USE_CUDA
      const size_t size = n * sizeof(T);

      if constexpr (m == MemorySpace::Host)
      {
         ptr = new T[n]{};
      }
      else
      {
         if constexpr (m == MemorySpace::Managed)
         {
            TENSOR_CHECK(cudaMallocManaged(&ptr, size) == cudaSuccess,
                         printf("cudaMallocManaged failed to allocate %zu bytes\n", size));
         }
         else if constexpr (m == MemorySpace::Device)
         {
            TENSOR_CHECK(cudaMalloc(&ptr, size) == cudaSuccess,
                         printf("cudaMalloc failed to allocate %zu bytes\n", size));
         }

         TENSOR_CHECK(cudaMemset(ptr, 0, size) == cudaSuccess,
                      printf("cudaMemset failed to initialize %zu bytes\n", size));
      }
#else
      ptr = new T[n]{};
#endif

#ifdef TENSOR_DEBUG_PRINT_MALLOC
      printf("Allocating %zu bytes at %p in %s memory space\n", n * sizeof(T), static_cast<void *>(ptr),
             memorySpaceToString(m));
#endif

      return ptr;
   }

   /// @brief Deallocates an array on the specified memory space. Returns nullptr.
   template <typename T, MemorySpace m = MemorySpace::Host>
   inline T *deallocate(T *ptr)
   {
      static_assert(m != MemorySpace::Unspecified, "Cannot deallocate memory in Unspecified memory space.");

      if (ptr == nullptr)
         return nullptr;

#ifdef TENSOR_DEBUG_PRINT_MALLOC
      printf("Deallocating memory at %p in %s memory space\n", static_cast<void *>(ptr), memorySpaceToString(m));
#endif

#ifdef TENSOR_USE_CUDA
      if constexpr (m == MemorySpace::Host)
      {
         delete[] ptr;
      }
      else if constexpr (m == MemorySpace::Device || m == MemorySpace::Managed)
      {
         TENSOR_CHECK(cudaFree(ptr) == cudaSuccess,
                      printf("cudaFree failed to free memory at %p\n", static_cast<void *>(ptr)));
      }
#else
      delete[] ptr;
#endif
      return nullptr;
   }

   /// @brief Allocator for STL containers.
   template <typename T, MemorySpace m = MemorySpace::Host>
   class allocator
   {
      static_assert(m != MemorySpace::Unspecified, "Cannot use allocator with Unspecified memory space.");

   public:
      using value_type = T;
      using pointer = T *;
      using size_type = size_t;

      static constexpr MemorySpace memory_space = m;

      allocator() noexcept = default;
      allocator(const allocator &) noexcept = default;
      allocator &operator=(const allocator &) noexcept = default;
      allocator(allocator &&) noexcept = default;
      allocator &operator=(allocator &&) noexcept = default;
      ~allocator() noexcept = default;

      static inline pointer allocate(size_type n)
      {
         return tensor::allocate<T, m>(n);
      }

      static inline void deallocate(pointer p, size_type)
      {
         tensor::deallocate<T, m>(p);
      }

      constexpr bool operator==(const allocator &)
      {
         return true;
      }

      constexpr bool operator!=(const allocator &)
      {
         return false;
      }
   };

   /// @brief Deleter for unique pointers.
   template <typename T, MemorySpace m = MemorySpace::Host>
   class deleter
   {
      static_assert(m != MemorySpace::Unspecified, "Cannot use deleter with Unspecified memory space.");

   public:
      static constexpr MemorySpace memory_space = m;

      constexpr deleter() = default;
      constexpr deleter(const deleter &) = default;
      constexpr deleter &operator=(const deleter &) = default;

      inline void operator()(T *ptr) const
      {
         tensor::deallocate<T, m>(ptr);
      }
   };

#ifdef TENSOR_USE_CUDA
   /**
    * @brief Synchronizes memory between different memory spaces. Namely, managed memory is prefetched to host or
    * device. If from is not Managed and to != from, an error is raised.
    */
   template <typename T>
   inline void synchronizeMemory(T *ptr, size_t n, MemorySpace from, MemorySpace to)
   {
      if (n == 0 || ptr == nullptr || from == to || to == MemorySpace::Managed)
         return;

      TENSOR_CHECK(to != MemorySpace::Unspecified,
                   printf("synchronizeMemory: cannot synchronize to Unspecified memory space\n"));

      TENSOR_CHECK(from == MemorySpace::Managed,
                   printf("synchronizeMemory: unsupported memory synchronization from %s to %s\n",
                          memorySpaceToString(from), memorySpaceToString(to)));

      size_t size = n * sizeof(T);
      cudaMemLocationType to_type = (to == MemorySpace::Host) ? cudaMemLocationTypeHost : cudaMemLocationTypeDevice;
      TENSOR_CHECK(cudaMemPrefetchAsync(ptr, size, to_type, 0) == cudaSuccess,
                   printf("cudaMemPrefetchAsync failed to prefetch %zu bytes\n", size));
   }
#else
   /**
    * @brief Synchronizes memory between different memory spaces. No-op in non-CUDA builds.
    */
   template <typename T>
   inline void synchronizeMemory(T *, size_t, MemorySpace, MemorySpace)
   {
   }
#endif
} // namespace tensor

namespace tensor::details
{
   /// @brief Copies n elements from src to dst. This is aware of the memory space of the pointers.
   template <typename T>
   inline void copy_n(const T *src, const size_t n, T *dst)
   {
      if (n == 0)
         return;
      if (src == dst)
         return;

      TENSOR_CHECK(src != nullptr && dst != nullptr, printf("tensor::copy_n: src and dst must be non-null\n"));

#ifdef TENSOR_USE_CUDA
      TENSOR_CHECK(cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyDefault) == cudaSuccess,
                   printf("cudaMemcpy failed to copy %zu bytes\n", n * sizeof(T)));
#else
      std::copy_n(src, n, dst);
#endif
   }

   inline constexpr bool compatibleMemorySpaces(MemorySpace a, MemorySpace b)
   {
#ifdef TENSOR_USE_CUDA
      return (a == MemorySpace::Managed || b == MemorySpace::Managed || a == b) ||
             (a == MemorySpace::Unspecified || b == MemorySpace::Unspecified);
#else
      return (a == MemorySpace::Host || a == MemorySpace::Unspecified) &&
             (b == MemorySpace::Host || b == MemorySpace::Unspecified);
#endif
   }
} // namespace tensor::details

#ifdef TENSOR_USE_CUDA
#ifdef TENSOR_DEVICE_CODE
#define TENSOR_DEBUG_VERIFY_MEMORY_SPACE(mem_space)            \
   TENSOR_DEBUG_ASSERT(mem_space != tensor::MemorySpace::Host, \
                       printf("Attempting to access Host memory from Device code.\n"))
#else
#define TENSOR_DEBUG_VERIFY_MEMORY_SPACE(mem_space)              \
   TENSOR_DEBUG_ASSERT(mem_space != tensor::MemorySpace::Device, \
                       printf("Attempting to access Device memory from Host code.\n"))
#endif
#else
#define TENSOR_DEBUG_VERIFY_MEMORY_SPACE(mem_space) // not needed
#endif

#ifdef TENSOR_USE_CUDA
#define TENSOR_REQUIRES_NOT_DEVICE_SPACE(mem_space) requires(mem_space != tensor::MemorySpace::Device)
#else
#define TENSOR_REQUIRES_NOT_DEVICE_SPACE(mem_space)
#endif