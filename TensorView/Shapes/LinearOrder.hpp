#pragma once

namespace tensor
{
   enum class LinearOrder
   {
      F, // Fortran order (column-major)
      C  // C order (row-major)
   };
} // namespace tensor