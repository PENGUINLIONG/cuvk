#pragma once
#include "cuvk/comdef.hpp"
#include "cuvk/storage.hpp"
#include "cuvk/pipeline.hpp"

L_CUVK_BEGIN_

// Dispatch deformation of bacteria using selected physical device.
class Deformation : public ComputeShaderContextual {
private:
  VkCommandBuffer _com_buf;

  bool create_com_buf();

public:
  std::vector<VkDescriptorSetLayoutBinding> layout_bindings() const override;
  Spirv comp_spirv() const override;

  bool execute(const   Storage& deform_specs, size_t n_deform_spec,
               L_INOUT Storage& bacs,         size_t n_bacs);
};
// Collision detection.
class CollisionDetection {
  bool execute(const Storage& bacs,     size_t n_bacs,
               L_OUT Storage& coll_map);
};
// Resolve elastic physics between cells, i.e. collisions.
class ElasticPhysics {
  bool execute(const   Storage& coll_map,
               L_INOUT Storage& bacs);
};
// Dispatch evaluation of universe images using selected physical device.
class Evaluation : public GraphicsShaderContextual {
  std::vector<VkDescriptorSetLayoutBinding> layout_bindings() const override;
  Spirv vert_spirv() const override;
  Spirv geom_spirv() const override;
  Spirv frag_spirv() const override;

  bool execute(const Storage& bacs,     size_t n_bacs,
               L_OUT Storage& diff_map,
               L_OUT Storage& costs);
};

L_CUVK_END_
