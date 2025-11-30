#pragma once
#include <format>
#include <iostream>

#include "TensorView.hpp"

class ColorText
{
public:
   static std::string red(const std::string &text)
   {
      return std::format("\033[31m{}\033[0m", text);
   }

   static std::string green(const std::string &text)
   {
      return std::format("\033[32m{}\033[0m", text);
   }
};

#define PRINT_RESULT(n_fails)                                                                           \
   if (n_fails == 0)                                                                                    \
   {                                                                                                    \
      std::cout << ColorText::green(std::format("{}: All tests passed!", __FILE__)) << std::endl;       \
   }                                                                                                    \
   else                                                                                                 \
   {                                                                                                    \
      std::cout << ColorText::red(std::format("{}: {} tests failed!", __FILE__, n_fails)) << std::endl; \
   }
