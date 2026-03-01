#pragma once
#include "TensorView/Expressions/ExpressionTraits.hpp"
#include "TensorView/Macros.hpp"
#include "TensorView/Tensors/makeTensor.hpp"

namespace tensor
{
   /**
    * @brief Evaluates an expression and returns a new tensor containing the result.
    *
    * This function materializes a lazy-evaluated expression into a concrete tensor.
    * The resulting tensor has the same shape as the expression and the value type is
    * determined by the expression's value_type.
    *
    * @tparam Expr The expression type (must satisfy the Expression concept)
    * @param expr The expression to evaluate
    * @return A new tensor containing the evaluated elements
    *
    * @example
    * Tensor<float, 2> a(3, 3), b(3, 3);
    * // ... fill a and b ...
    * auto result = eval(a + b);  // Materializes a + b into a tensor
    */
   template <details::Expression Expr>
   auto eval(const Expr &expr)
   {
      auto result = makeTensorLike(expr);
      result = expr;
      return result;
   }

} // namespace tensor
