#pragma once
#include <array>

#include "TensorView/Expressions/ExpressionBase.hpp"

namespace tensor::details
{
   /**
    * @brief ScalarExpr wraps a scalar value as a 0-dimensional expression.
    * This allows scalars to participate in expression templates, particularly
    * useful for broadcasting and fancy indexing where one operand may be a scalar
    * and the other a view.
    *
    * @tparam T The scalar value type
    */
   template <typename T>
   class ScalarExpr : public ExpressionBase<ScalarExpr<T>>
   {
   private:
      T _value;

   public:
      using value_type = T;

      /**
       * @brief Construct a ScalarExpr from a value.
       */
      explicit ScalarExpr(T value) : _value(value) {}

      /**
       * @brief Returns the memory space.
       * Scalars are treated as Unspecified so they can safely participate with
       * expressions in any concrete memory space.
       */
      static constexpr MemorySpace memorySpace()
      {
         return MemorySpace::Unspecified;
      }

      /**
       * @brief Returns 0 since this is a scalar (0-dimensional).
       */
      static constexpr index_t numDims()
      {
         return 0;
      }

      /**
       * @brief Returns 1 since a scalar has size 1.
       */
      constexpr index_t size() const
      {
         return 1;
      }

      /**
       * @brief For a scalar, shape is always 1 for any dimension.
       */
      constexpr index_t shape(index_t /*dim*/) const
      {
         return 1;
      }

      /**
       * @brief Element access with linear index.
       * For a scalar, always returns the value regardless of index.
       */
      decltype(auto) operator[](index_t /*index*/) const
      {
         return _value;
      }

      /**
       * @brief Multi-dimensional element access.
       * For a scalar with no indices, returns the value.
       */
      decltype(auto) at() const
      {
         return _value;
      }

      /**
       * @brief Multi-dimensional element access with indices.
       * For a scalar, always returns the value regardless of indices.
       */
      template <typename... Indices>
      decltype(auto) at(Indices &&.../*indices*/) const
      {
         return _value;
      }

      /**
       * @brief Multi-dimensional element access with a std::array of indices.
       */
      decltype(auto) at(const std::array<index_t, 0> & /*multi_index*/) const
      {
         return _value;
      }

      /**
       * @brief Returns a view of the scalar.
       * For a scalar, this is just itself.
       */
      auto view() const
      {
         return *this;
      }

      /**
       * @brief Returns a raw view of the scalar.
       * For a scalar, this is just itself.
       */
      auto raw() const
      {
         return *this;
      }

      /**
       * @brief Scalar expressions don't support iteration.
       * This is a placeholder that returns nullptr.
       */
      auto begin() const
      {
         return static_cast<const T *>(nullptr);
      }

      /**
       * @brief Scalar expressions don't support iteration.
       * This is a placeholder that returns nullptr.
       */
      auto end() const
      {
         return static_cast<const T *>(nullptr);
      }
   };

   /**
    * @brief Helper function to create a ScalarExpr.
    */
   template <typename T>
   auto makeScalarExpr(T value)
   {
      return ScalarExpr<T>(value);
   }

} // namespace tensor::details
