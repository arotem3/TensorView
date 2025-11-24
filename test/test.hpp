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