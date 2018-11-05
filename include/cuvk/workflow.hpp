#pragma once
#include "cuvk/comdef.hpp"
#include "cuvk/storage.hpp"
#include "cuvk/pipeline.hpp"

L_CUVK_BEGIN_

// Dispatch deformation of bacteria using selected physical device.
class Deformation : public ComputeShaderContextual {
private:
  static std::vector<LayoutBinding> _layout_binds;

public:
  Deformation() = default;
  const std::vector<LayoutBinding>& layout_binds() const override;

  Spirv comp_spirv() const override;

  bool execute(const StorageBufferView& deform_specs,
               const StorageBufferView& bacs,
               L_OUT StorageBufferView& bacs_out);
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
private:
  static std::vector<LayoutBinding> _layout_binds;
  static std::vector<Binding> _in_binds;
  static std::vector<Attribute> _in_attrs;

public:
  const std::vector<LayoutBinding>& layout_binds() const override;
  const std::vector<Binding>& in_binds() const override;
  const std::vector<Attribute>& in_attrs() const override;

  Spirv vert_spirv() const override;
  Spirv geom_spirv() const override;
  Spirv frag_spirv() const override;

  bool execute(const Storage& bacs,     size_t n_bacs,
               L_OUT Storage& diff_map,
               L_OUT Storage& costs);
};

L_CUVK_END_
