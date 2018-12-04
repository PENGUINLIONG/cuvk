#pragma once
#include "cuvk/comdef.hpp"

L_CUVK_BEGIN_

L_STATIC struct VulkanRequirements {
  static constexpr uint32_t
    vulkan_version = VK_MAKE_VERSION(1,0,0);
  static constexpr const char*
    app_name = "L_CUVK";
  static constexpr uint32_t
    app_version = VK_MAKE_VERSION(0,0,1);
};

const uint32_t MAX_DEV_QUEUE_COUNT = 8;
const uint32_t MAX_GRAPH_PIPE_STAGE_COUNT = 5;
const float DEFAULT_QUEUE_PRIORITY = 0.5;


L_CUVK_END_
