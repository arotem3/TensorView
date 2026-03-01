#pragma once
#include <type_traits>

#include "TensorView/Expressions/ExpressionBase.hpp"

namespace tensor::details
{
   /**
    * @brief Type trait to detect if a type is an expression.
    *
    * A type is considered an expression if it inherits from ExpressionBase<T>.
    * This trait can be used in SFINAE contexts and requires clauses to enable
    * or disable function overloads based on whether a type is an expression.
    *
    * @tparam T The type to check
    */
   template <typename T>
   struct IsExpression : std::false_type
   {};

   /**
    * @brief Specialization for types that derive from ExpressionBase.
    */
   template <typename T>
      requires std::is_base_of_v<ExpressionBase<T>, T>
   struct IsExpression<T> : std::true_type
   {};

   /**
    * @brief Helper variable template for IsExpression.
    */
   template <typename T>
   inline constexpr bool is_expression = IsExpression<T>::value;

   /**
    * @brief Concept to constrain template parameters to expression types.
    */
   template <typename T>
   concept Expression = is_expression<std::remove_cvref_t<T>>;

   /**
    * @brief Type trait to extract the value type from an expression.
    *
    * For expressions that have a value_type member, this trait provides
    * access to the underlying element type of the expression.
    *
    * @tparam Expr The expression type
    */
   template <Expression Expr>
   struct ExpressionValueType
   {
      using type = typename std::remove_cvref_t<Expr>::value_type;
   };

   /**
    * @brief Helper alias template for ExpressionValueType.
    */
   template <Expression Expr>
   using expression_value_type_t = typename ExpressionValueType<Expr>::type;

   /**
    * @brief Type trait to get the number of dimensions of an expression.
    *
    * @tparam Expr The expression type
    */
   template <Expression Expr>
   struct ExpressionNumDims
   {
      static constexpr index_t value = std::remove_cvref_t<Expr>::numDims();
   };

   /**
    * @brief Helper variable template for ExpressionNumDims.
    */
   template <Expression Expr>
   inline constexpr index_t expression_num_dims = ExpressionNumDims<Expr>::value;

} // namespace tensor::details
