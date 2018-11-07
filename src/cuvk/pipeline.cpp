#include "cuvk/pipeline.hpp"
#include "cuvk/logger.hpp"
#include <map>

L_CUVK_BEGIN_

Spirv::Spirv() : _code() {}
Spirv::Spirv(std::vector<uint32_t>&& code) :
  _code(std::forward<std::vector<uint32_t>>(code)) {}
Spirv::Spirv(const char* spirv, size_t len) : _code() {
  _code.resize((len + 3) / 4);
  // TODO: Optimize `memcpy` later.
  std::memcpy((void*)_code.data(), spirv, len);
}

size_t Spirv::size() const {
  return _code.size() * sizeof(uint32_t);
}
const uint32_t* Spirv::data() const {
  return _code.data();
}

bool PipelineContextual::create_shader_module(
    const Spirv& spv, L_OUT VkShaderModule& mod) const {
  VkShaderModuleCreateInfo smci{};
  smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  smci.codeSize = spv.size();
  smci.pCode = spv.data();

  if (L_VK <- vkCreateShaderModule(ctxt().dev(), &smci, nullptr, &mod)) {
    LOG.error("unable to instantiate shader module");
    return false;
  }
  return true;
}

bool PipelineContextual::create_pl() {
  auto dev = ctxt().dev();

  auto dslbs = layout_binds();
  VkDescriptorSetLayoutCreateInfo dslci{};
  dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dslci.pBindings = dslbs.data();
  dslci.bindingCount = (uint32_t)dslbs.size();

  if (L_VK <- vkCreateDescriptorSetLayout(
      dev, &dslci, nullptr, &_desc_set_layout)) {
    LOG.error("unable to create descriptor set layout");
    return false;
  }

  VkPipelineLayoutCreateInfo plci{};
  plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &_desc_set_layout;

  if (L_VK <- vkCreatePipelineLayout(dev, &plci, nullptr, &_pl_layout)) {
    LOG.error("unable to create pipeline layout");
    return false;
  }

  // Create descriptor sets.

  // Collect the number of each type of descriptors.
  std::map<VkDescriptorType, uint32_t> desc_count_map;
  for (const auto& bind : dslbs) {
    auto it = desc_count_map.find(bind.descriptorType);
    if (it == desc_count_map.end()) {
      desc_count_map.emplace(std::make_pair(bind.descriptorType, bind.descriptorCount));
    } else {
      it->second += bind.descriptorCount;
    }
  }
  std::vector<VkDescriptorPoolSize> dpss;
  for (const auto& pair : desc_count_map) {
    dpss.emplace_back(VkDescriptorPoolSize{ pair.first, pair.second });
  }

  VkDescriptorPoolCreateInfo dpci {};
  dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpci.maxSets = 1;
  dpci.poolSizeCount = static_cast<uint32_t>(dpss.size());
  dpci.pPoolSizes = dpss.data();

  if (L_VK <- vkCreateDescriptorPool(dev, &dpci, nullptr, &_desc_pool)) {
    LOG.error("unable to create descriptor pool");
    return false;
  }

  VkDescriptorSetAllocateInfo dsai {};
  dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsai.descriptorPool = _desc_pool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &_desc_set_layout;

  if (L_VK <- vkAllocateDescriptorSets(dev, &dsai, &_desc_set)) {
    LOG.error("unable to allocate descriptor sets");
    return false;
  }

  return true;
}
void PipelineContextual::destroy_pl() {
  auto dev = ctxt().dev();

  // Once `_desc_set` is allocated, it doesn't need to be released, as we didn't
  // turn the `VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT` flag on.
  _desc_set = VK_NULL_HANDLE;
  if (_desc_pool) {
    vkDestroyDescriptorPool(dev, _desc_pool, nullptr);
    _desc_pool = VK_NULL_HANDLE;
  }
  if (_pl_layout) {
    vkDestroyPipelineLayout(dev, _pl_layout, nullptr);
    _pl_layout = VK_NULL_HANDLE;
  }
  if (_desc_set_layout) {
    vkDestroyDescriptorSetLayout(dev, _desc_set_layout, nullptr);
    _desc_set_layout = VK_NULL_HANDLE;
  }
}

bool PipelineContextual::context_changing() {
  destroy_pl();
  return true;
}
bool PipelineContextual::context_changed() {
  return create_pl();
}



bool ComputeShaderContextual::create_pl() {
  if (!PipelineContextual::create_pl()) {
    return false;
  }
  if (!create_shader_module(comp_spirv(), _mod)) {
    return false;
  }
  VkComputePipelineCreateInfo cpci{};
  cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  cpci.layout = _pl_layout;
  cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  cpci.stage.module = _mod;
  cpci.stage.pName = "main";

  // TODO: (penguinliong) Use cache.
  VkPipelineCache pl_cache = VK_NULL_HANDLE;
  if (L_VK <- vkCreateComputePipelines(ctxt().dev(),
    pl_cache, 1, &cpci, nullptr, &_pl)) {
    LOG.error("unable to create compute pipeline");
    return false;
  }
  return true;
}
void ComputeShaderContextual::destroy_pl() {
  auto dev = ctxt().dev();

  if (_pl) {
    vkDestroyPipeline(dev, _pl, nullptr);
    _pl = VK_NULL_HANDLE;
  }
  if (_mod) {
    vkDestroyShaderModule(dev, _mod, nullptr);
    _mod = VK_NULL_HANDLE;
  }
  PipelineContextual::destroy_pl();
}



GraphicsShaderContextual::GraphicsShaderContextual(
  uint32_t rows, uint32_t cols, uint32_t layers) :
  _rows(rows), _cols(cols), _layers(layers) { }

VkRenderPass GraphicsShaderContextual::render_pass() const {
  return _pass;
}

bool GraphicsShaderContextual::create_pl() {
  auto dev = ctxt().dev();

  if (!PipelineContextual::create_pl()) {
    return false;
  }

  // Load shader stages.
  if (!create_shader_module(vert_spirv(), _mods[0]) ||
      !create_shader_module(geom_spirv(), _mods[1]) ||
      !create_shader_module(frag_spirv(), _mods[2])) {
    return false;
  }

  std::array<VkPipelineShaderStageCreateInfo, 3> pssci{};
  pssci[0].sType = pssci[1].sType = pssci[2].sType =
    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pssci[0].pName = pssci[1].pName = pssci[2].pName = "main";
  pssci[0].module = _mods[0];
  pssci[1].module = _mods[1];
  pssci[2].module = _mods[2];
  pssci[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  pssci[1].stage = VK_SHADER_STAGE_GEOMETRY_BIT;
  pssci[2].stage = VK_SHADER_STAGE_FRAGMENT_BIT;

  _pass = create_pass();
  if (_pass == VK_NULL_HANDLE) return false;

  // Vertex inputs.
  auto vibds = in_binds();
  auto viads = in_attrs();

  VkPipelineVertexInputStateCreateInfo pvisci{};
  pvisci.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  pvisci.vertexAttributeDescriptionCount = (uint32_t)viads.size();
  pvisci.pVertexAttributeDescriptions = viads.data();
  pvisci.vertexBindingDescriptionCount = (uint32_t)vibds.size();
  pvisci.pVertexBindingDescriptions = vibds.data();

  // Input assembly.
  VkPipelineInputAssemblyStateCreateInfo piasci{};
  piasci.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  piasci.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;

  // Viewport and scissors.
  VkViewport viewport { 0, 0, _cols, _rows, 0.0, 0.0 };
  VkRect2D scissor { { 0, 0 }, { _cols, _rows } };

  VkPipelineViewportStateCreateInfo pvsci{};
  pvsci.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  pvsci.viewportCount = 1;
  pvsci.pViewports = &viewport;
  pvsci.scissorCount = 1;
  pvsci.pScissors = &scissor;

  // Resterization requirements.
  VkPipelineRasterizationStateCreateInfo prsci{};
  prsci.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  prsci.cullMode = VK_CULL_MODE_BACK_BIT;
  prsci.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  prsci.lineWidth = 1.;

  // Multisampling which we don't need.
  VkPipelineMultisampleStateCreateInfo pmsci{};
  pmsci.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  pmsci.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  pmsci.minSampleShading = 1.0;

  // Blending which we don't need.
  auto blends = out_blends();
  VkPipelineColorBlendStateCreateInfo pcbsci{};
  pcbsci.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  pcbsci.blendConstants[0] = 1.0;
  pcbsci.attachmentCount = blends.size();
  pcbsci.pAttachments = blends.data();

  // Create graphics pipeline.
  VkGraphicsPipelineCreateInfo gpci{};
  gpci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  gpci.layout = _pl_layout;
  gpci.stageCount = 3;
  gpci.pStages = pssci.data();
  gpci.pVertexInputState = &pvisci;
  gpci.pInputAssemblyState = &piasci;
  gpci.pViewportState = &pvsci;
  gpci.pRasterizationState = &prsci;
  gpci.pMultisampleState = &pmsci;
  gpci.pColorBlendState = &pcbsci;
  gpci.renderPass = _pass;
  gpci.subpass = 0;

  // TODO: (penguinliong) Do we need to use cache?
  if (L_VK <-vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1,
      &gpci, nullptr, &_pl)) {
    LOG.error("unable to create graphics pipeline");
    return false;
  }
  return true;
}
void GraphicsShaderContextual::destroy_pl() {
  auto dev = ctxt().dev();

  if (_pl) {
    vkDestroyPipeline(dev, _pl, nullptr);
    _pl = VK_NULL_HANDLE;
  }

  for (auto mod = _mods.begin(); mod != _mods.end(); ++mod) {
    if (*mod) {
      vkDestroyShaderModule(dev, *mod, nullptr);
      *mod = VK_NULL_HANDLE;
    }
  }
  PipelineContextual::destroy_pl();
}

L_CUVK_END_
