#pragma once
#include "TensorView/Macros.hpp"
#include "TensorView/Utility/Memory.hpp"

namespace tensor::details
{
   /**
    * @brief Base class for all expression types using CRTP (Curiously Recurring Template Pattern).
    *
    * This class provides a minimal interface that all expressions must satisfy:
    * - Indexing via operator[]
    * - Shape information via shape(), size(), numDims()
    * - Iterator support via begin(), end()
    *
    * Expression templates allow lazy evaluation of tensor operations, enabling
    * optimizations like loop fusion and elimination of temporary objects.
    *
    * @tparam Derived The derived expression type (CRTP parameter)
    */
   template <typename Derived>
   class ExpressionBase
   {
   public:
      /**
       * @brief Access the derived expression object.
       */
      constexpr const Derived &derived() const
      {
         return static_cast<const Derived &>(*this);
      }

      /**
       * @brief Access the derived expression object.
       */
      constexpr Derived &derived()
      {
         return static_cast<Derived &>(*this);
      }

      /**
       * @brief Returns the number of dimensions of the expression.
       * Delegates to the derived class implementation.
       */
      static constexpr index_t numDims()
      {
         return Derived::numDims();
      }

      /**
       * @brief Returns the memory space of the expression.
       * Delegates to the derived class implementation.
       */
      static constexpr MemorySpace memorySpace()
      {
         return Derived::memorySpace();
      }

      /**
       * @brief Returns the total number of elements in the expression.
       * Delegates to the derived class implementation.
       */
      constexpr index_t size() const
      {
         return derived().size();
      }

      /**
       * @brief Returns the size of the specified dimension.
       * Delegates to the derived class implementation.
       */
      constexpr index_t shape(index_t dim) const
      {
         return derived().shape(dim);
      }

      /**
       * @brief Element access with linear index.
       * Delegates to the derived class implementation.
       */
      decltype(auto) operator[](index_t index)
      {
         return derived()[index];
      }

      /**
       * @brief Element access with linear index (const version).
       * Delegates to the derived class implementation.
       */
      decltype(auto) operator[](index_t index) const
      {
         return derived()[index];
      }

      /**
       * @brief Multi-dimensional element access with indices.
       * For fancy index expressions, the behavior depends on the derived class implementation.
       * Delegates to the derived class implementation.
       */
      template <typename... Indices>
      decltype(auto) at(Indices &&...indices)
      {
         return derived().at(std::forward<Indices>(indices)...);
      }

      /**
       * @brief Multi-dimensional element access with indices (const version).
       * For fancy index expressions, the behavior depends on the derived class implementation.
       * Delegates to the derived class implementation.
       */
      template <typename... Indices>
      decltype(auto) at(Indices &&...indices) const
      {
         return derived().at(std::forward<Indices>(indices)...);
      }

      /**
       * @brief Multi-dimensional element access operator with indices.
       * For fancy index expressions, the behavior depends on the derived class implementation.
       * Delegates to the derived class implementation via at().
       */
      template <typename... Indices>
      decltype(auto) operator()(Indices &&...indices)
      {
         return at(std::forward<Indices>(indices)...);
      }

      /**
       * @brief Multi-dimensional element access operator with indices (const version).
       * For fancy index expressions, the behavior depends on the derived class implementation.
       * Delegates to the derived class implementation via at().
       */
      template <typename... Indices>
      decltype(auto) operator()(Indices &&...indices) const
      {
         return at(std::forward<Indices>(indices)...);
      }

      /**
       * @brief Returns an iterator to the beginning of the expression.
       * Delegates to the derived class implementation.
       */
      auto begin()
      {
         return derived().begin();
      }

      /**
       * @brief Returns a const iterator to the beginning of the expression.
       * Delegates to the derived class implementation.
       */
      auto begin() const
      {
         return derived().begin();
      }

      /**
       * @brief Returns an iterator to the end of the expression.
       * Delegates to the derived class implementation.
       */
      auto end()
      {
         return derived().end();
      }

      /**
       * @brief Returns a const iterator to the end of the expression.
       * Delegates to the derived class implementation.
       */
      auto end() const
      {
         return derived().end();
      }

      /**
       * @brief Returns a persistent/reference view of the expression with the same data.
       * Delegates to the derived class implementation.
       */
      auto view()
      {
         return derived().view();
      }

      /**
       * @brief Returns a const persistent/reference view of the expression with the same data.
       * Delegates to the derived class implementation.
       */
      auto view() const
      {
         return derived().view();
      }

      /**
       * @brief Returns a raw view of the expression with the same data.
       * Raw views do not guarantee data lifetime; the user is responsible for ensuring
       * the underlying data remains valid.
       * Delegates to the derived class implementation.
       */
      auto raw()
      {
         return derived().raw();
      }

      /**
       * @brief Returns a const raw view of the expression with the same data.
       * Raw views do not guarantee data lifetime; the user is responsible for ensuring
       * the underlying data remains valid.
       * Delegates to the derived class implementation.
       */
      auto raw() const
      {
         return derived().raw();
      }

   protected:
      // Protected constructors prevent direct instantiation of ExpressionBase
      ExpressionBase() = default;
      ~ExpressionBase() = default;
      ExpressionBase(const ExpressionBase &) = default;
      ExpressionBase(ExpressionBase &&) = default;
      ExpressionBase &operator=(const ExpressionBase &) = default;
      ExpressionBase &operator=(ExpressionBase &&) = default;
   };
} // namespace tensor::details
