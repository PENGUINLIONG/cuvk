#include "cuvk/pipeline.hpp"
#include "cuvk/context.hpp"
#include "cuvk/logger.hpp"
#include <map>

L_CUVK_BEGIN_

ShaderStage::operator VkPipelineShaderStageCreateInfo() const noexcept {
  return {
    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr,
    0, stage, shader->shader, entry
  };
}

ShaderStage Shader::stage(
  L_STATIC const char* entry, VkShaderStageFlagBits stage) const noexcept {
  return { this, entry, stage };
}

const Shader& ShaderManager::declare_shader(
  std::vector<uint32_t> spv) noexcept {
  return shaders.emplace_back(Shader{ std::move(spv), VK_NULL_HANDLE });
}

ShaderManager::ShaderManager(const Context& ctxt) noexcept :
  ctxt(&ctxt) {}
bool ShaderManager::make(bool keep_spv) noexcept {
  LOG.trace("making managed shader modules");
  for (auto& shader : shaders) {
    if (shader.shader) { continue; }
    if (shader.spv.empty()) {
      LOG.warning("source spv is empty; ensure you are not remaking a dropped "
        "shader manager that was made with `keep_spv` set false");
      continue;
    }

    VkShaderModuleCreateInfo smci {};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = shader.spv.size() * sizeof(uint32_t);
    smci.pCode = shader.spv.data();

    if (L_VK <- vkCreateShaderModule(
      ctxt->dev, &smci, nullptr, &shader.shader)) {
      LOG.error("unable to create shader module");
      return false;
    }
    if (!keep_spv) {
      shader.spv.resize(0);
      shader.spv.shrink_to_fit();
    }
  }
  return true;
}
void ShaderManager::drop() noexcept {
  LOG.trace("dropping managed shader modules");
  for (auto& shader : shaders) {
    if (shader.shader) {
      vkDestroyShaderModule(ctxt->dev, shader.shader, nullptr);
      shader.shader = VK_NULL_HANDLE;
    }
  }
  shaders.clear();
}
ShaderManager::~ShaderManager() noexcept { drop(); }

ShaderManager::ShaderManager(ShaderManager&& right) :
  ctxt(right.ctxt),
  shaders(std::exchange(right.shaders, {})) {}



DescriptorSet& DescriptorSet::write(
  uint32_t bind_pt, const BufferSlice& buf_slice,
  VkDescriptorType desc_type) noexcept {
  VkDescriptorBufferInfo dbi {
    buf_slice.buf_alloc->buf, buf_slice.offset, buf_slice.size
  };
  VkWriteDescriptorSet wds {
    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
    desc_set, bind_pt, 0,
    1, desc_type,
    nullptr, &dbi, nullptr,
  };
  vkUpdateDescriptorSets(ctxt->dev, 1, &wds, 0, nullptr);
  return *this;
}
DescriptorSet& DescriptorSet::write(
  uint32_t bind_pt, const ImageView& img_view, VkImageLayout layout,
  VkDescriptorType desc_type) noexcept {
  VkDescriptorImageInfo dii {
    VK_NULL_HANDLE, // TODO: (penguinliong) Use sampler.
    img_view.img_view,
    layout,
  };
  VkWriteDescriptorSet wds {
    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr,
    desc_set, bind_pt, 0,
    1, desc_type,
    &dii, nullptr, nullptr,
  };
  vkUpdateDescriptorSets(ctxt->dev, 1, &wds, 0, nullptr);
  return *this;
}



std::optional<DescriptorSet> GraphicsPipeline::desc_set() noexcept {
  DescriptorSet rv(*ctxt, *this);
  if (rv.make()) {
    return rv;
  } else {
    return {};
  }
}

std::optional<DescriptorSet> ComputePipeline::desc_set() noexcept {
  DescriptorSet rv(*ctxt, *this);
  if (rv.make()) {
    return rv;
  } else {
    return {};
  }
}



const GraphicsPipeline& PipelineManager::declare_graph_pipe(
  const char* name,
  L_STATIC Span<ShaderStage> stages,
  L_STATIC Span<VkPushConstantRange> push_const_rngs,
  L_STATIC Span<VkDescriptorSetLayoutBinding> layout_binds,
  L_STATIC Span<VkVertexInputBindingDescription> in_binds,
  L_STATIC Span<VkVertexInputAttributeDescription> in_attrs,
  L_STATIC Span<VkPipelineColorBlendAttachmentState> out_blends,
  L_STATIC Span<VkAttachmentDescription> out_attach_descs,
  L_STATIC Span<VkAttachmentReference> out_attach_refs,
  const ImageView& attach) noexcept {
  if (stages.size() > MAX_GRAPH_PIPE_STAGE_COUNT) {
    LOG.error("graphics pipeline must not have more than 5 stages");
    std::terminate();
  }
  GraphicsPipeline pipe {
    ctxt,
    name,
    stages, push_const_rngs,
    { in_binds, in_attrs, out_blends, out_attach_descs,
      out_attach_refs },
    { layout_binds },
    VK_NULL_HANDLE, VK_NULL_HANDLE,
    VK_NULL_HANDLE, VK_NULL_HANDLE,
    &attach,
  };

  return graph_pipes.emplace_back(std::move(pipe));
}
const ComputePipeline& PipelineManager::declare_comp_pipe(
  const char* name,
  const ShaderStage* stage,
  L_STATIC Span<VkPushConstantRange> push_const_rngs,
  L_STATIC Span<VkDescriptorSetLayoutBinding> layout_binds,
  std::optional<std::array<uint32_t, 3>> local_workgrp_size) noexcept {
  ComputePipeline pipe {
    ctxt,
    name,
    stage, push_const_rngs,
    local_workgrp_size,
    {},
    VK_NULL_HANDLE, VK_NULL_HANDLE
  };
  pipe.desc_set_layout.layout_binds = std::move(layout_binds);

  return comp_pipes.emplace_back(std::move(pipe));
}

PipelineManager::PipelineManager(const Context& ctxt) noexcept :
  ctxt(&ctxt), graph_pipes(), comp_pipes() {}
bool PipelineManager::make() noexcept {
  LOG.trace("making managed pipelines");
  return make_graph_pipes() && make_comp_pipes();
}
void PipelineManager::drop() noexcept {
  LOG.trace("dropping managed pipelines");
  drop_comp_pipes();
  drop_graph_pipes();
}
PipelineManager::~PipelineManager() noexcept { drop(); }

PipelineManager::PipelineManager(PipelineManager&& right) noexcept :
  ctxt(right.ctxt),
  graph_pipes(std::move(right.graph_pipes), {}),
  comp_pipes(std::move(right.comp_pipes), {}) {}

bool PipelineManager::make_layouts(
  L_STATIC Span<VkDescriptorSetLayoutBinding> layout_binds,
  L_STATIC Span<VkPushConstantRange> push_const_rngs,
  L_OUT VkDescriptorSetLayout& desc_set_layout,
  L_OUT VkPipelineLayout& pipe_layout) noexcept {

  VkDescriptorSetLayoutCreateInfo dslci{};
  dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dslci.pBindings = layout_binds.data();
  dslci.bindingCount = static_cast<uint32_t>(layout_binds.size());

  if (L_VK <- vkCreateDescriptorSetLayout(
      ctxt->dev, &dslci, nullptr, &desc_set_layout)) {
    LOG.error("unable to create descriptor set layout");
    return false;
  }

  VkPipelineLayoutCreateInfo plci{};
  plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &desc_set_layout;
  plci.pushConstantRangeCount = static_cast<uint32_t>(push_const_rngs.size());
  plci.pPushConstantRanges = push_const_rngs.data();

  if (L_VK <- vkCreatePipelineLayout(ctxt->dev, &plci, nullptr, &pipe_layout)) {
    LOG.error("unable to create pipeline layout");
    return false;
  }
  return true;
}
std::vector<VkDescriptorPoolSize> make_desc_pool_sizes(
  Span<VkDescriptorSetLayoutBinding> layout_binds) {
  // Collect the number of each type of descriptors.
  std::map<VkDescriptorType, uint32_t> desc_count_map;
  for (const auto& bind : layout_binds) {
    auto it = desc_count_map.find(bind.descriptorType);
    if (it == desc_count_map.end()) {
      desc_count_map.emplace(
        std::make_pair(bind.descriptorType, bind.descriptorCount));
    } else {
      it->second += bind.descriptorCount;
    }
  }
  std::vector<VkDescriptorPoolSize> dpss;
  dpss.reserve(desc_count_map.size());
  for (const auto& pair : desc_count_map) {
    dpss.emplace_back(VkDescriptorPoolSize{ pair.first, pair.second });
  }
  return std::move(dpss);
}
bool PipelineManager::make_graph_pipes() noexcept {
  for (auto& pipe : graph_pipes) {
    if (pipe.pipe) { continue; }
    if (!make_layouts(pipe.desc_set_layout.layout_binds, pipe.push_const_rngs,
      pipe.desc_set_layout.desc_set_layout,
      pipe.pipe_layout)) {
      return false;
    }

    /* Render pass. */ {
      auto& attach_refs = pipe.graph_io_req.out_attach_refs;

      VkSubpassDescription sd {};
      sd.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
      sd.colorAttachmentCount = static_cast<uint32_t>(attach_refs.size());
      sd.pColorAttachments = attach_refs.data();

      auto& attach_descs = pipe.graph_io_req.out_attach_descs;

      VkRenderPassCreateInfo rpci {};
      rpci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
      rpci.attachmentCount = static_cast<uint32_t>(attach_descs.size());
      rpci.pAttachments = attach_descs.data();
      rpci.subpassCount = 1;
      rpci.pSubpasses = &sd;

      if (L_VK <- vkCreateRenderPass(ctxt->dev, &rpci, nullptr, &pipe.pass)) {
        LOG.error("unable to create render pass for graphics pipeline '{}'",
          pipe.name);
        return false;
      }
    }

    /* Framebuffer. */ {
      // TODO: (penguinliong) Support multi attachments. Check with the input of
      // render pass creation.
      auto img_view = pipe.attach->img_view;
      VkFramebufferCreateInfo fci {};
      fci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
      fci.renderPass = pipe.pass;
      fci.attachmentCount = 1;
      fci.pAttachments = &img_view;
      fci.width = pipe.attach->img_slice.img_alloc->req.extent.width;
      fci.height = pipe.attach->img_slice.img_alloc->req.extent.height;
      fci.layers = pipe.attach->img_slice.nlayer.value_or(1);

      if (L_VK <- vkCreateFramebuffer(ctxt->dev, &fci, nullptr, &pipe.framebuf)) {
        LOG.error("unable to create framebuffer");
        return false;
      }
    }

    auto& vibds = pipe.graph_io_req.in_binds;
    auto& viads = pipe.graph_io_req.in_attrs;

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
    // Viewport info is update on each draw. We can ignore the viewport info as
    // we have dynamic viewport state.
    VkViewport viewport {
      0, 0,
      static_cast<float>(pipe.attach->img_slice.img_alloc->req.extent.width),
      static_cast<float>(pipe.attach->img_slice.img_alloc->req.extent.height),
      0.0, 0.0,
    };
    VkRect2D scissor { { 0, 0 }, pipe.attach->img_slice.img_alloc->req.extent };

    VkPipelineViewportStateCreateInfo pvsci{};
    pvsci.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    pvsci.viewportCount = 1;
    pvsci.pViewports = &viewport;
    pvsci.scissorCount = 1;
    pvsci.pScissors = &scissor;

    // Resterization requirements.
    VkPipelineRasterizationStateCreateInfo prsci{};
    prsci.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    prsci.cullMode = VK_CULL_MODE_NONE;
    prsci.polygonMode = VK_POLYGON_MODE_FILL;
    prsci.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    prsci.lineWidth = 1.;

    // Multisampling which we don't need.
    VkPipelineMultisampleStateCreateInfo pmsci{};
    pmsci.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    pmsci.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    pmsci.minSampleShading = 1.0;

    // Blending which we don't need.
    auto& blends = pipe.graph_io_req.out_blends;
    VkPipelineColorBlendStateCreateInfo pcbsci{};
    pcbsci.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    pcbsci.blendConstants[0] = 1.0;
    pcbsci.attachmentCount = static_cast<uint32_t>(blends.size());
    pcbsci.pAttachments = blends.data();

    uint32_t nstage = 0;
    std::array<VkPipelineShaderStageCreateInfo, 5> psscis{};
    for (auto& stage : pipe.stages) {
      psscis[nstage++] = stage;
    }

    // Create graphics pipeline.
    VkGraphicsPipelineCreateInfo gpci{};
    gpci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gpci.layout = pipe.pipe_layout;
    gpci.stageCount = nstage;
    gpci.pStages = psscis.data();
    gpci.pVertexInputState = &pvisci;
    gpci.pInputAssemblyState = &piasci;
    gpci.pViewportState = &pvsci;
    gpci.pRasterizationState = &prsci;
    gpci.pMultisampleState = &pmsci;
    gpci.pColorBlendState = &pcbsci;
    gpci.renderPass = pipe.pass;
    gpci.subpass = 0;

    // TODO: (penguinliong) Do we need to use cache?
    if (L_VK <-vkCreateGraphicsPipelines(ctxt->dev, VK_NULL_HANDLE, 1,
        &gpci, nullptr, &pipe.pipe)) {
      LOG.error("unable to create graphics pipeline '{}'", pipe.name);
      return false;
    }
    LOG.info("created graphics pipeline '{}'", pipe.name);

    pipe.desc_set_layout.desc_pool_sizes =
      make_desc_pool_sizes(pipe.desc_set_layout.layout_binds);
  }
  return true;
}
bool PipelineManager::make_comp_pipes() noexcept {
  for (auto& pipe : comp_pipes) {
    if (pipe.pipe) { continue; }
    if (!make_layouts(pipe.desc_set_layout.layout_binds, pipe.push_const_rngs,
      pipe.desc_set_layout.desc_set_layout,
      pipe.pipe_layout)) {
      LOG.error("unable to create layouts for compute pipeline '{}'",
        pipe.name);
      return false;
    }

    VkSpecializationInfo si {};
    std::array<VkSpecializationMapEntry, 3> smes {};

    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.layout = pipe.pipe_layout;
    cpci.stage = *pipe.stage;

    if (pipe.local_workgrp_size.has_value()) {
      smes[0].constantID = 1;
      smes[0].size = sizeof(uint32_t);
      smes[0].offset = 0;

      smes[1].constantID = 2;
      smes[1].size = sizeof(uint32_t);
      smes[1].offset = sizeof(uint32_t);

      smes[2].constantID = 3;
      smes[2].size = sizeof(uint32_t);
      smes[2].offset = 2 * sizeof(uint32_t);

      si.mapEntryCount = static_cast<uint32_t>(smes.size());
      si.pMapEntries = smes.data();
      si.dataSize = sizeof(uint32_t) * pipe.local_workgrp_size->size();
      si.pData = pipe.local_workgrp_size->data();

      LOG.info("compute pipeline '{}' has specialized its workgroups to "
        "({}, {}, {})", pipe.name,
        pipe.local_workgrp_size->at(0),
        pipe.local_workgrp_size->at(1),
        pipe.local_workgrp_size->at(2));
      cpci.stage.pSpecializationInfo = &si;
    }

    if (L_VK <- vkCreateComputePipelines(ctxt->dev,
      VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe.pipe)) {
      LOG.error("unable to create compute pipeline '{}'", pipe.name);
      return false;
    }
    LOG.info("created compute pipeline '{}'", pipe.name);

    pipe.desc_set_layout.desc_pool_sizes =
      make_desc_pool_sizes(pipe.desc_set_layout.layout_binds);
  }

  return true;
}
void PipelineManager::drop_graph_pipes() noexcept {
  for (auto& pipe : graph_pipes) {
    if (pipe.pipe) {
      vkDestroyPipeline(ctxt->dev, pipe.pipe, nullptr);
      pipe.pipe = VK_NULL_HANDLE;
    }
    if (pipe.pass) {
      vkDestroyRenderPass(ctxt->dev, pipe.pass, nullptr);
      pipe.pass = VK_NULL_HANDLE;
    }
    if (pipe.pipe_layout) {
      vkDestroyPipelineLayout(ctxt->dev, pipe.pipe_layout, nullptr);
      pipe.pipe_layout = VK_NULL_HANDLE;
    }
    if (pipe.desc_set_layout.desc_set_layout) {
      vkDestroyDescriptorSetLayout(
        ctxt->dev, pipe.desc_set_layout.desc_set_layout, nullptr);
      pipe.desc_set_layout.desc_set_layout = VK_NULL_HANDLE;
    }
    if (pipe.framebuf) {
      vkDestroyFramebuffer(ctxt->dev, pipe.framebuf, nullptr);
      pipe.framebuf = VK_NULL_HANDLE;
    }
  }
  LOG.info("dropped all {} graphics pipelines", graph_pipes.size());
  graph_pipes.clear();
}
void PipelineManager::drop_comp_pipes() noexcept {
  for (auto& pipe : comp_pipes) {
    if (pipe.pipe) {
      vkDestroyPipeline(ctxt->dev, pipe.pipe, nullptr);
      pipe.pipe = VK_NULL_HANDLE;
    }
    if (pipe.pipe_layout) {
      vkDestroyPipelineLayout(ctxt->dev, pipe.pipe_layout, nullptr);
      pipe.pipe_layout = VK_NULL_HANDLE;
    }
    if (pipe.desc_set_layout.desc_set_layout) {
      vkDestroyDescriptorSetLayout(
        ctxt->dev, pipe.desc_set_layout.desc_set_layout, nullptr);
      pipe.desc_set_layout.desc_set_layout = VK_NULL_HANDLE;
    }
  }
  LOG.info("dropped all {} compute pipelines", comp_pipes.size());
  comp_pipes.clear();
}



DescriptorSet::DescriptorSet(
  const Context& ctxt,
  const GraphicsPipeline& pipe) noexcept :
  ctxt(&ctxt),
  desc_set_layout(&pipe.desc_set_layout),
  desc_pool(),
  desc_set() {}
DescriptorSet::DescriptorSet(
  const Context& ctxt,
  const ComputePipeline& pipe) noexcept :
  ctxt(&ctxt),
  desc_set_layout(&pipe.desc_set_layout),
  desc_pool(),
  desc_set() {}

bool DescriptorSet::make() noexcept {
  if (desc_pool) { return true; }
  VkDescriptorPoolCreateInfo dpci {};
  dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpci.maxSets = 1;
  dpci.poolSizeCount =
    static_cast<uint32_t>(desc_set_layout->desc_pool_sizes.size());
  dpci.pPoolSizes = desc_set_layout->desc_pool_sizes.data();

  if (L_VK <- vkCreateDescriptorPool(ctxt->dev, &dpci, nullptr, &desc_pool)) {
    LOG.error("unable to create descriptor pool");
    return false;
  }

  VkDescriptorSetAllocateInfo dsai {};
  dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsai.descriptorPool = desc_pool;
  dsai.descriptorSetCount = 1;
  dsai.pSetLayouts = &desc_set_layout->desc_set_layout;

  if (L_VK <- vkAllocateDescriptorSets(ctxt->dev, &dsai, &desc_set)) {
    LOG.error("unable to allocate descriptor sets");
    return false;
  }
  return true;
}
void DescriptorSet::drop()  noexcept{
  if (desc_pool) {
    vkDestroyDescriptorPool(ctxt->dev, desc_pool, nullptr);
    desc_pool = VK_NULL_HANDLE;
    desc_set = VK_NULL_HANDLE;
  }
}
DescriptorSet::~DescriptorSet() noexcept { drop(); }

DescriptorSet::DescriptorSet(DescriptorSet&& right) noexcept :
  ctxt(right.ctxt),
  desc_set_layout(right.desc_set_layout),
  desc_pool(right.desc_pool),
  desc_set(right.desc_set) {}

L_CUVK_END_
