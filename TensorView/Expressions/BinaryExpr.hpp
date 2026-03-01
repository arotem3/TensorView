#pragma once
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

#include "TensorView/Access/Iterator.hpp"
#include "TensorView/Expressions/ExpressionBase.hpp"
#include "TensorView/Expressions/ExpressionTraits.hpp"
#include "TensorView/Expressions/ScalarExpr.hpp"
#include "TensorView/Macros.hpp"
#include "TensorView/Shapes/CompareShapes.hpp"

namespace tensor::details
{
   template <typename Op, typename LHS, typename RHS>
      requires(Expression<LHS> && Expression<RHS>)
   class BinaryExpr : public ExpressionBase<BinaryExpr<Op, LHS, RHS>>
   {
   public:
      using lhs_type = LHS;
      using rhs_type = RHS;
      using op_type = Op;
      using lhs_value_type = typename LHS::value_type;
      using rhs_value_type = typename RHS::value_type;
      using value_type = std::common_type_t<lhs_value_type, rhs_value_type>;

   private:
      LHS _lhs;
      RHS _rhs;
      Op _op;

   public:
      static constexpr index_t numDims()
      {
         return std::max(LHS::numDims(), RHS::numDims());
      }

      static constexpr MemorySpace memorySpace()
      {
         constexpr MemorySpace lhs_space = LHS::memorySpace();
         constexpr MemorySpace rhs_space = RHS::memorySpace();

         if constexpr (lhs_space == rhs_space)
            return lhs_space;

         if constexpr (lhs_space == MemorySpace::Unspecified)
            return rhs_space;
         if constexpr (rhs_space == MemorySpace::Unspecified)
            return lhs_space;

#ifdef TENSOR_USE_CUDA
         if constexpr (lhs_space == MemorySpace::Host || rhs_space == MemorySpace::Host)
            return MemorySpace::Host;

         if constexpr (lhs_space == MemorySpace::Device || rhs_space == MemorySpace::Device)
            return MemorySpace::Device;

         static_assert(lhs_space == MemorySpace::Managed && rhs_space == MemorySpace::Managed,
                       "Unhandled memory-space combination in BinaryExpr::memorySpace().");
         return MemorySpace::Managed;
#else
         return MemorySpace::Host;
#endif
      }

      explicit BinaryExpr(LHS lhs, RHS rhs, Op op = Op()) : _lhs(std::move(lhs)), _rhs(std::move(rhs)), _op(op)
      {
         static_assert(compatibleMemorySpaces(LHS::memorySpace(), RHS::memorySpace()),
                       "BinaryExpr operands must have compatible memory spaces.");
         TENSOR_REQUIRE_EQUAL_SHAPES(_lhs, _rhs);
      }

      constexpr index_t size() const
      {
         index_t total = 1;
         for (index_t i = 0; i < numDims(); ++i)
            total *= shape(i);
         return total;
      }

      constexpr index_t shape(index_t dim) const
      {
         return (dim < _lhs.numDims()) ? _lhs.shape(dim) : _rhs.shape(dim);
      }

      decltype(auto) operator[](index_t index) const
      {
         return _op(static_cast<value_type>(_lhs[index]), static_cast<value_type>(_rhs[index]));
      }

      template <typename... Indices>
      decltype(auto) at(Indices &&...indices) const
         requires(sizeof...(Indices) > 0 &&
                  (sizeof...(Indices) != 1 ||
                   !std::is_same_v<std::remove_cvref_t<std::tuple_element_t<0, std::tuple<Indices...>>>,
                                   std::array<index_t, numDims()>>))
      {
         static_assert(sizeof...(Indices) == numDims(), "Number of indices must match expression dimensions");

         auto indicesTuple = std::forward_as_tuple(std::forward<Indices>(indices)...);

         auto makeView = [&](auto &&expr) -> decltype(auto) {
            constexpr index_t exprDims = std::remove_cvref_t<decltype(expr)>::numDims();

            if constexpr (exprDims == numDims())
               return expr.at(std::forward<Indices>(indices)...);
            else // select leading indices
               return [&]<std::size_t... Is>(std::index_sequence<Is...>) -> decltype(auto) {
                  return expr.at(std::get<Is>(indicesTuple)...);
               }(std::make_index_sequence<exprDims>{});
         };

         auto lhsResult = makeView(_lhs);
         auto rhsResult = makeView(_rhs);

         using lhsResultT = decltype(lhsResult);
         using rhsResultT = decltype(rhsResult);
         using lhsBareT = std::remove_cvref_t<lhsResultT>;
         using rhsBareT = std::remove_cvref_t<rhsResultT>;

         constexpr bool lhsIsExpr = is_expression<lhsBareT>;
         constexpr bool rhsIsExpr = is_expression<rhsBareT>;

         if constexpr (lhsIsExpr && rhsIsExpr)
         {
            return BinaryExpr<Op, lhsBareT, rhsBareT>(std::move(lhsResult), std::move(rhsResult), _op);
         }
         else if constexpr (lhsIsExpr && !rhsIsExpr)
         {
            auto rhsScalarExpr = ScalarExpr<rhsBareT>(rhsResult);
            return BinaryExpr<Op, lhsBareT, ScalarExpr<rhsBareT>>(std::move(lhsResult), std::move(rhsScalarExpr), _op);
         }
         else if constexpr (!lhsIsExpr && rhsIsExpr)
         {
            auto lhsScalarExpr = ScalarExpr<lhsBareT>(lhsResult);
            return BinaryExpr<Op, ScalarExpr<lhsBareT>, rhsBareT>(std::move(lhsScalarExpr), std::move(rhsResult), _op);
         }
         else
         {
            return _op(static_cast<value_type>(lhsResult), static_cast<value_type>(rhsResult));
         }
      }

      decltype(auto) at(const std::array<index_t, numDims()> &multi_index) const
      {
         auto getValue = [&](auto &&expr) -> decltype(auto) {
            constexpr index_t exprDims = std::remove_cvref_t<decltype(expr)>::numDims();

            if constexpr (exprDims == numDims())
               return expr.at(multi_index);
            else // select leading indices
               return expr.at([&]<std::size_t... Is>(std::index_sequence<Is...>) {
                  return std::array<index_t, exprDims>{multi_index[Is]...};
               }(std::make_index_sequence<exprDims>{}));
         };

         return _op(static_cast<value_type>(getValue(_lhs)), static_cast<value_type>(getValue(_rhs)));
      }

      auto begin() const
      {
         return TensorBegin(*this);
      }

      auto end() const
      {
         return TensorEndSentinel{};
      }

      auto view() const
      {
         return *this;
      }

      auto raw() const
      {
         return BinaryExpr<Op, decltype(_lhs.raw()), decltype(_rhs.raw())>(_lhs.raw(), _rhs.raw(), _op);
      }
   };

} // namespace tensor::details
