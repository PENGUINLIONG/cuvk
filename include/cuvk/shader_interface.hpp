#pragma once
#include "comdef.hpp"
#include <array>
#include <vulkan/vulkan.h>

L_CUVK_BEGIN_

namespace shader_interface {
// -------------------------------------
// * Definitions are in STD430 layout. *
// -------------------------------------

struct DeformSpecs {
  std::array<float, 2> translate;
  std::array<float, 2> stretch;
  float rotate;
  int32_t _pad0;
};
static_assert(sizeof(DeformSpecs) == 24);

struct Bacterium {
  std::array<float, 2> pos;
  std::array<float, 2> size;
  float orient;
  uint32_t univ;
};
static_assert(sizeof(Bacterium) == 24);

}

L_CUVK_END_
