#pragma once
#include <functional>
#include <utility>

#include "TensorView/Expressions/BinaryExpr.hpp"
#include "TensorView/Expressions/ExpressionTraits.hpp"
#include "TensorView/Utility/ToDevice.hpp"

namespace tensor::details
{
   template <typename Op, Expression LHS, Expression RHS>
   auto makeBinaryExpr(const ExpressionBase<LHS> &lhs, const ExpressionBase<RHS> &rhs)
   {
#ifdef TENSOR_USE_CUDA
      auto lhs_view = lhs.view();
      auto rhs_view = rhs.view();

      constexpr MemorySpace lhs_space = std::remove_cvref_t<decltype(lhs_view)>::memorySpace();
      constexpr MemorySpace rhs_space = std::remove_cvref_t<decltype(rhs_view)>::memorySpace();

      if constexpr (lhs_space == MemorySpace::Host && rhs_space == MemorySpace::Host)
      {
         return BinaryExpr<Op, std::remove_cvref_t<decltype(lhs_view)>, std::remove_cvref_t<decltype(rhs_view)>>(
             std::move(lhs_view), std::move(rhs_view), Op{});
      }
      else
      {
         auto lhs_device = toDevice(std::move(lhs_view));
         auto rhs_device = toDevice(std::move(rhs_view));
         return BinaryExpr<Op, std::remove_cvref_t<decltype(lhs_device)>, std::remove_cvref_t<decltype(rhs_device)>>(
             std::move(lhs_device), std::move(rhs_device), Op{});
      }
#else
      auto lhs_view = lhs.view();
      auto rhs_view = rhs.view();
      return BinaryExpr<Op, std::remove_cvref_t<decltype(lhs_view)>, std::remove_cvref_t<decltype(rhs_view)>>(
          std::move(lhs_view), std::move(rhs_view), Op{});
#endif
   }

   /**
    * @brief Addition operator for expressions.
    * Automatically handles memory space compatibility.
    */
   template <Expression LHS, Expression RHS>
   auto operator+(const ExpressionBase<LHS> &lhs, const ExpressionBase<RHS> &rhs)
   {
      return makeBinaryExpr<std::plus<>>(lhs, rhs);
   }

   /**
    * @brief Subtraction operator for expressions.
    * Automatically handles memory space compatibility.
    */
   template <Expression LHS, Expression RHS>
   auto operator-(const ExpressionBase<LHS> &lhs, const ExpressionBase<RHS> &rhs)
   {
      return makeBinaryExpr<std::minus<>>(lhs, rhs);
   }

   /**
    * @brief Multiplication operator for expressions (element-wise).
    * Automatically handles memory space compatibility.
    */
   template <Expression LHS, Expression RHS>
   auto operator*(const ExpressionBase<LHS> &lhs, const ExpressionBase<RHS> &rhs)
   {
      return makeBinaryExpr<std::multiplies<>>(lhs, rhs);
   }

   /**
    * @brief Division operator for expressions (element-wise).
    * Automatically handles memory space compatibility.
    */
   template <Expression LHS, Expression RHS>
   auto operator/(const ExpressionBase<LHS> &lhs, const ExpressionBase<RHS> &rhs)
   {
      return makeBinaryExpr<std::divides<>>(lhs, rhs);
   }

   template <Expression LHS, typename RHS>
      requires(!Expression<std::remove_cvref_t<RHS>>)
   auto operator+(const ExpressionBase<LHS> &lhs, RHS rhs)
   {
      return BinaryExpr<std::plus<>, decltype(lhs.view()), ScalarExpr<std::remove_cvref_t<RHS>>>(
          lhs.view(), ScalarExpr<std::remove_cvref_t<RHS>>(rhs), std::plus<>());
   }

   template <typename LHS, Expression RHS>
      requires(!Expression<std::remove_cvref_t<LHS>>)
   auto operator+(LHS lhs, const ExpressionBase<RHS> &rhs)
   {
      return BinaryExpr<std::plus<>, ScalarExpr<std::remove_cvref_t<LHS>>, decltype(rhs.view())>(
          ScalarExpr<std::remove_cvref_t<LHS>>(lhs), rhs.view(), std::plus<>());
   }

   template <Expression LHS, typename RHS>
      requires(!Expression<std::remove_cvref_t<RHS>>)
   auto operator-(const ExpressionBase<LHS> &lhs, RHS rhs)
   {
      return BinaryExpr<std::minus<>, decltype(lhs.view()), ScalarExpr<std::remove_cvref_t<RHS>>>(
          lhs.view(), ScalarExpr<std::remove_cvref_t<RHS>>(rhs), std::minus<>());
   }

   template <typename LHS, Expression RHS>
      requires(!Expression<std::remove_cvref_t<LHS>>)
   auto operator-(LHS lhs, const ExpressionBase<RHS> &rhs)
   {
      return BinaryExpr<std::minus<>, ScalarExpr<std::remove_cvref_t<LHS>>, decltype(rhs.view())>(
          ScalarExpr<std::remove_cvref_t<LHS>>(lhs), rhs.view(), std::minus<>());
   }

   template <Expression LHS, typename RHS>
      requires(!Expression<std::remove_cvref_t<RHS>>)
   auto operator*(const ExpressionBase<LHS> &lhs, RHS rhs)
   {
      return BinaryExpr<std::multiplies<>, decltype(lhs.view()), ScalarExpr<std::remove_cvref_t<RHS>>>(
          lhs.view(), ScalarExpr<std::remove_cvref_t<RHS>>(rhs), std::multiplies<>());
   }

   template <typename LHS, Expression RHS>
      requires(!Expression<std::remove_cvref_t<LHS>>)
   auto operator*(LHS lhs, const ExpressionBase<RHS> &rhs)
   {
      return BinaryExpr<std::multiplies<>, ScalarExpr<std::remove_cvref_t<LHS>>, decltype(rhs.view())>(
          ScalarExpr<std::remove_cvref_t<LHS>>(lhs), rhs.view(), std::multiplies<>());
   }

   template <Expression LHS, typename RHS>
      requires(!Expression<std::remove_cvref_t<RHS>>)
   auto operator/(const ExpressionBase<LHS> &lhs, RHS rhs)
   {
      return BinaryExpr<std::divides<>, decltype(lhs.view()), ScalarExpr<std::remove_cvref_t<RHS>>>(
          lhs.view(), ScalarExpr<std::remove_cvref_t<RHS>>(rhs), std::divides<>());
   }

   template <typename LHS, Expression RHS>
      requires(!Expression<std::remove_cvref_t<LHS>>)
   auto operator/(LHS lhs, const ExpressionBase<RHS> &rhs)
   {
      return BinaryExpr<std::divides<>, ScalarExpr<std::remove_cvref_t<LHS>>, decltype(rhs.view())>(
          ScalarExpr<std::remove_cvref_t<LHS>>(lhs), rhs.view(), std::divides<>());
   }

} // namespace tensor::details

namespace tensor::operators
{
   using details::operator+;
   using details::operator-;
   using details::operator*;
   using details::operator/;
} // namespace tensor::operators
