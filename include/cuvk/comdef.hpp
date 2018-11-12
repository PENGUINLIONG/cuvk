#pragma once
#include <cstdint>
#include <optional>

#define L_CUVK_BEGIN_ namespace cuvk {
#define L_CUVK_END_ }

// The decorated variable is an output variable.
#define L_OUT
// The value of decorated variable will be taken as input and a new value will
// be set as output later.
#define L_INOUT
// The decorated variable is required to have static lifetime.
#define L_STATIC
