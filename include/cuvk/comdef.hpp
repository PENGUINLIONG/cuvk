#pragma once
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
// Vectors are not used in resource managers because we need to keep all the
// references alive. Vectors change elements' addresses on reallocation.
#include <list>
#include <vulkan/vulkan.h>

#define L_CUVK_BEGIN_ namespace cuvk {
#define L_CUVK_END_ }

// The decorated variable is an output variable.
#define L_OUT
// The value of decorated variable will be taken as input and a new value will
// be set as output later.
#define L_INOUT
// The decorated variable is required to have static lifetime.
#define L_STATIC

#ifdef NDEBUG
#define L_DEBUG false
#else
#define L_DEBUG true
#endif
