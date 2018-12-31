#include "cuvk/cuvk.h"
#include "cuvk/comdef.hpp"
#include "cuvk/context.hpp"
#include "cuvk/storage.hpp"
#include "cuvk/pipeline.hpp"
#include "cuvk/executor.hpp"
#include "cuvk/logger.hpp"
#include "cuvk/shader_interface.hpp"
#include <atomic>
#include <mutex>
#include <fstream>
#include <future>

using namespace cuvk;
using namespace cuvk::shader_interface;

Vulkan vk;
std::mutex sync;
std::string phys_dev_json;

const std::array<VkQueueFlags, 1> CUVK_QUEUE_CAPS = {
  VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT,
};
constexpr VkPhysicalDeviceFeatures cuvk_phys_dev_feat() {
  VkPhysicalDeviceFeatures feat {};
  feat.geometryShader = true;
  feat.shaderStorageBufferArrayDynamicIndexing = true;
  feat.shaderStorageImageArrayDynamicIndexing = true;
  return feat;
}
const VkPhysicalDeviceFeatures CUVK_PHYS_DEV_FEAT = cuvk_phys_dev_feat();




// Find the last power of 2 that is contained by the given value.
template<typename T>
constexpr T last_pow_2(T a) {
  auto i = sizeof(T) * 8;
  while (i-- && a >> i == 0) {}
  return 1 << i;
}
// Find the a power of 2 that can contain the given value.
template<typename T>
constexpr T next_pow_2(T a) {
  for (int i = 0; i < sizeof(T) * 8; ++i) {
    if ((1 << i) > a) {
      return i << i;
    }
  }
}





std::vector<uint32_t> read_spirv(const std::string& path) {
  std::ifstream f("assets/shaders/" + path + ".spv",
    std::ios::in | std::ios::binary | std::ios::ate);
  if (!f) {
    LOG.error("unable to read spirv: {}", path);
    return {};
  }
  size_t size = f.tellg();
  std::vector<uint32_t> buf;
  buf.resize(size / sizeof (uint32_t));
  f.seekg(std::ios::beg);
  f.read(reinterpret_cast<char*>(buf.data()), size);
  f.close();

  return buf;
}

struct CuvkDeformPipeline {
  const Shader& comp;

  std::array<ShaderStage, 1> stages;
  std::array<VkPushConstantRange, 1> push_const_rngs;
  std::array<VkDescriptorSetLayoutBinding, 3> desc_layout_binds;

  std::optional<std::array<uint32_t, 3>> local_workgrp;
  const ComputePipeline& pipe;

  CuvkDeformPipeline(const CuvkMemoryRequirements mem_req,
    ShaderManager& shader_mgr, PipelineManager& pipe_mgr) :
    comp(shader_mgr.declare_shader(read_spirv("deform.comp"))),
    stages({
      comp.stage("main", VK_SHADER_STAGE_COMPUTE_BIT),
    }),
    push_const_rngs({
      VkPushConstantRange
      { VK_SHADER_STAGE_COMPUTE_BIT, 0, 12 },
    }),
    desc_layout_binds({
      VkDescriptorSetLayoutBinding
      // DeformSpecs[] deform_specs
      { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
      // Bac[] bacs
      { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
      // Bac[] bacs
      { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
    }),
    pipe(pipe_mgr.declare_comp_pipe("deform",
      PipelineRequirements { stages, push_const_rngs, desc_layout_binds },
      ComputePipelineRequirements {}
    )) {}
};
struct CuvkEvalPipeline {
  const Shader& vert;
  const Shader& geom;
  const Shader& frag;

  std::array<ShaderStage, 3> stages;
  std::array<VkPushConstantRange, 1> push_const_rngs;
  std::array<VkDescriptorSetLayoutBinding, 0> desc_layout_binds;

  std::array<VkVertexInputBindingDescription, 1> vert_binds;
  std::array<VkVertexInputAttributeDescription, 4> vert_attrs;
  VkExtent2D viewport;
  std::array<VkAttachmentDescription, 1> attach_descs;
  std::array<VkAttachmentReference, 1> attach_refs;
  std::array<VkPipelineColorBlendAttachmentState, 1> blends;

  const GraphicsPipeline& pipe;

  CuvkEvalPipeline(const CuvkMemoryRequirements& mem_req,
    ShaderManager& shader_mgr, PipelineManager& pipe_mgr) :
    vert(shader_mgr.declare_shader(read_spirv("eval.vert"))),
    geom(shader_mgr.declare_shader(read_spirv("eval.geom"))),
    frag(shader_mgr.declare_shader(read_spirv("eval.frag"))),
    stages({
      vert.stage("main", VK_SHADER_STAGE_VERTEX_BIT),
      geom.stage("main", VK_SHADER_STAGE_GEOMETRY_BIT),
      frag.stage("main", VK_SHADER_STAGE_FRAGMENT_BIT),
    }),
    push_const_rngs({
      VkPushConstantRange
      { VK_SHADER_STAGE_GEOMETRY_BIT, 0, 4 },
    }),

    vert_binds({
      VkVertexInputBindingDescription
      { 0, 6 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX }, // Bacterium
    }),
    vert_attrs ({
      VkVertexInputAttributeDescription
      { 0, 0, VK_FORMAT_R32G32_SFLOAT, 0                    }, // pos
      { 1, 0, VK_FORMAT_R32G32_SFLOAT, 2 * sizeof(float)    }, // size
      { 2, 0, VK_FORMAT_R32_SFLOAT,    4 * sizeof(float)    }, // orient
      { 3, 0, VK_FORMAT_R32_UINT,      5 * sizeof(uint32_t) }, // univ
    }),
    viewport({ mem_req.width, mem_req.height }),
    attach_descs({
      VkAttachmentDescription {
        0, VK_FORMAT_R32_SFLOAT, VK_SAMPLE_COUNT_1_BIT,
        VK_ATTACHMENT_LOAD_OP_CLEAR,
        VK_ATTACHMENT_STORE_OP_STORE,
        VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        VK_ATTACHMENT_STORE_OP_DONT_CARE,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      },
    }),
    attach_refs({
      VkAttachmentReference
      { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL },
    }),
    blends({
      VkPipelineColorBlendAttachmentState
      { true,
        VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ONE, VK_BLEND_OP_MAX,
        VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ONE, VK_BLEND_OP_MAX,
        0xF,
      },
    }),
    pipe(pipe_mgr.declare_graph_pipe("eval",
      PipelineRequirements { stages, push_const_rngs, desc_layout_binds },
      GraphicsPipelineRequirements {
        vert_binds, vert_attrs, viewport, attach_descs, attach_refs, blends
      })) {}
};
struct CuvkCostPipeline {
  const Shader& comp;

  struct Scheduling {
    // Number of packs in a universe.
    uint32_t npack_univ;
    // Number of sections (excluding the residual section).
    uint32_t nsec;
    // Number of sections (including the residual section).
    uint32_t nsec_actual;
    // The number of local workgroups.
    uint32_t npack_sec;
    // If there is no residual, `npack_res` will be marked as 0.
    uint32_t npack_res;

    Scheduling(const CuvkMemoryRequirements& mem_req,
      const VkPhysicalDeviceLimits& limits) {
      // Number of packs in a universe. A pack means 4 touching pixels in a row.
      npack_univ = mem_req.width * mem_req.height / 4;
      // Number of packs in each section.
      // The Vulkan specification didn't claim that the limit
      // `maxComputeWorkGroupSize[0]` must be less than or equal to
      // `maxComputeWorkGroupInvocations`. Although most vendors who still have
      // sanity obey such rule, some platforms still wanna say ____ you. Under
      // most scenario `maxComputeWorkGroupInvocations` can be divided exactly
      // by 32. See GPUInfo for the full list of limit values:
      //   vulkan.gpuinfo.org/displaydevicelimit.php
      npack_sec = std::min(
        limits.maxComputeWorkGroupInvocations,
        limits.maxComputeWorkGroupSize[0]);
      // Number of sections.
      nsec = npack_univ / npack_sec;
      // Number of packs in the residual.
      npack_res = npack_univ % npack_sec;
      // Number of sections + residual (if present).
      nsec_actual = (npack_res > 0) ? nsec + 1 : nsec;
    }
  } scheduling;

  std::array<ShaderStage, 1> stages;
  std::array<VkDescriptorSetLayoutBinding, 4> desc_layout_binds;
  std::array<VkPushConstantRange, 1> push_const_rngs;

  const ComputePipeline& pipe_sec;
  // Pipeline for residuals when the number of pixels (in unit of `vec4`) is not
  // a multiple of `maxComputeWorkGroupInvocations`.
  const ComputePipeline& pipe_res;

  CuvkCostPipeline(const CuvkMemoryRequirements& mem_req,
    ShaderManager& shader_mgr, PipelineManager& pipe_mgr) :
    comp(shader_mgr.declare_shader(read_spirv("cost.comp"))),
    scheduling(mem_req,
      pipe_mgr.ctxt->req.phys_dev_info->phys_dev_props.limits),
    stages({
      comp.stage("main", VK_SHADER_STAGE_COMPUTE_BIT),
    }),
    desc_layout_binds({
      VkDescriptorSetLayoutBinding
      // image2D real_univ
      { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
      // image2DArray sim_univs
      { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
      // float[] temp
      { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
      // float[] costs
      { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    }),
    push_const_rngs({
      VkPushConstantRange
      { VK_SHADER_STAGE_COMPUTE_BIT, 0, 16 },
    }),
    pipe_sec(pipe_mgr.declare_comp_pipe("cost_sec",
      PipelineRequirements { stages, push_const_rngs, desc_layout_binds },
      ComputePipelineRequirements {
        std::array<uint32_t, 3> { scheduling.npack_sec, 1, 1 }
      })),
    pipe_res(pipe_mgr.declare_comp_pipe("cost_res",
      PipelineRequirements { stages, push_const_rngs, desc_layout_binds },
      ComputePipelineRequirements {
        std::array<uint32_t, 3> { scheduling.npack_res, 1, 1 }
      })) {}
};

struct CuvkPipelines {
  ShaderManager shader_mgr;
  PipelineManager pipe_mgr;

  CuvkDeformPipeline deform_pipe;
  CuvkEvalPipeline eval_pipe;
  CuvkCostPipeline cost_pipe;

  CuvkPipelines(const Context& ctxt, const CuvkMemoryRequirements& mem_req) :
    shader_mgr(ctxt),
    pipe_mgr(ctxt),

    deform_pipe(mem_req, shader_mgr, pipe_mgr),
    eval_pipe(mem_req, shader_mgr, pipe_mgr),
    cost_pipe(mem_req, shader_mgr, pipe_mgr) {
  }
  bool make() {
    return shader_mgr.make(false) && pipe_mgr.make();
  }
  void drop() {
    pipe_mgr.drop();
    shader_mgr.drop();
  }
  ~CuvkPipelines() {
    drop();
  }
};


//
// Allocations.
//


struct MemoryAllocationGuidelines {
  BufferSizer hv_buf_sizer;
  BufferSizer do_buf_sizer;
  ImageSizer do_img_sizer;
  struct {
    RawBufferSlice deform_specs;
    RawBufferSlice bacs;
    RawBufferSlice bacs_out;
  } deformation;
  struct {
    RawBufferSlice bacs;
    RawBufferSlice real_univ;
    RawImageSlice sim_univs_temps;
    RawBufferSlice sum_temp;
    RawBufferSlice sim_univs;
    RawBufferSlice partial_costs;
  } evaluation;

  MemoryAllocationGuidelines(const Context& ctxt, const CuvkPipelines& pipes,
    const CuvkMemoryRequirements& mem_req) {
    auto& limits = ctxt.req.phys_dev_info->phys_dev_props.limits;
    auto storage_buf_alignment = limits.minStorageBufferOffsetAlignment;
    deformation.deform_specs = hv_buf_sizer.allocate<DeformSpecs>(
      mem_req.nspec, storage_buf_alignment);
    deformation.bacs = hv_buf_sizer.allocate<Bacterium>(
      mem_req.nbac, storage_buf_alignment);
    deformation.bacs_out = hv_buf_sizer.allocate<Bacterium>(
      mem_req.nspec * mem_req.nbac, storage_buf_alignment);

    auto cost_sch = pipes.cost_pipe.scheduling;
    auto nsec = cost_sch.nsec_actual;
    auto univ_size = mem_req.width * mem_req.height;

    evaluation.bacs = hv_buf_sizer.allocate<Bacterium>(
      mem_req.nspec * mem_req.nbac, storage_buf_alignment);
    evaluation.real_univ = hv_buf_sizer.allocate<float>(
      univ_size, storage_buf_alignment);
    evaluation.sim_univs_temps = do_img_sizer.allocate(mem_req.nuniv);
    evaluation.sum_temp = do_buf_sizer.allocate<float>(
      mem_req.nuniv * univ_size / 4, storage_buf_alignment);
    evaluation.sim_univs = hv_buf_sizer.allocate<float>(
      mem_req.nuniv * univ_size, storage_buf_alignment);
    evaluation.partial_costs = hv_buf_sizer.allocate<float>(
      mem_req.nuniv * nsec, storage_buf_alignment);
  }
};

struct CuvkDeformationAllocations {
  // Direct inputs.
  BufferSlice deform_specs;
  BufferSlice bacs;
  // Direct outputs.
  BufferSlice bacs_out;
};
struct CuvkEvaluationAllocations {
  // Direct inputs.
  BufferSlice bacs;
  BufferSlice real_univ;
  // Intermediate memories.
  std::vector<ImageView> sim_univs_temps;
  std::vector<Framebuffer> sim_univs_temp_framebufs;
  ImageSlice sim_univs_temp_entire;
  BufferSlice sum_temp;
  // Direct outputs.
  BufferSlice sim_univs;
  BufferSlice partial_costs;
};

struct CuvkAllocations {
  HeapManager heap_mgr;

  const BufferAllocation& hv_buf;
  const BufferAllocation& do_buf;
  const ImageAllocation& do_img;

  CuvkDeformationAllocations deformation_allocs;
  CuvkEvaluationAllocations evaluation_allocs;

  std::vector<const ImageView*> framebuf_refs;

  CuvkAllocations(const Context& ctxt, const CuvkPipelines& pipes,
    const CuvkMemoryRequirements& mem_req,
    const MemoryAllocationGuidelines& req) :
    heap_mgr(ctxt),
    hv_buf(heap_mgr.declare_buf(req.hv_buf_sizer,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
      VK_BUFFER_USAGE_TRANSFER_DST_BIT |
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
      MemoryVisibility::HostVisible)),
    do_buf(heap_mgr.declare_buf(req.do_buf_sizer,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      MemoryVisibility::DeviceOnly)),
    do_img(heap_mgr.declare_img({ mem_req.width, mem_req.height },
      req.do_img_sizer, VK_FORMAT_R32_SFLOAT,
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
      VK_IMAGE_USAGE_TRANSFER_DST_BIT |
      VK_IMAGE_USAGE_STORAGE_BIT |
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
      VK_IMAGE_TILING_OPTIMAL,
      MemoryVisibility::DeviceOnly)),
    deformation_allocs({
      hv_buf.slice(req.deformation.deform_specs),
      hv_buf.slice(req.deformation.bacs),
      hv_buf.slice(req.deformation.bacs_out),
    }),
    evaluation_allocs({
      hv_buf.slice(req.evaluation.bacs),
      hv_buf.slice(req.evaluation.real_univ),
      {},
      {},
      do_img.slice(req.evaluation.sim_univs_temps, true),
      do_buf.slice(req.evaluation.sum_temp),
      hv_buf.slice(req.evaluation.sim_univs),
      hv_buf.slice(req.evaluation.partial_costs),
    }),
    framebuf_refs() {
    
    auto limits = ctxt.req.phys_dev_info->phys_dev_props.limits;
    // Number of framebuffers that use full support (max number of layers).
    auto nfull_framebuf = mem_req.nuniv / limits.maxFramebufferLayers;
    // Number of layers in the last framebuffer.
    auto nuniv_last_framebuf = mem_req.nuniv % limits.maxFramebufferLayers;
    // Number of framebuffer to be created.
    auto nframebuf =
      nuniv_last_framebuf == 0 ? nfull_framebuf : nfull_framebuf + 1;
    // Reserve spaces for framebuffers
    evaluation_allocs.sim_univs_temps.reserve(nframebuf);
    evaluation_allocs.sim_univs_temp_framebufs.reserve(nframebuf);

    // Create image views and framebuffers for each universe that has full
    // capacity.
    uint32_t univ_offset = 0;
    for (auto i = 0u; i < nfull_framebuf; ++i) {
      framebuf_refs.push_back(&evaluation_allocs.sim_univs_temps.emplace_back(
          do_img.view(univ_offset, limits.maxFramebufferLayers)));
      // Make framebuffer for each view.
      evaluation_allocs.sim_univs_temp_framebufs.emplace_back(
        ctxt, pipes.eval_pipe.pipe.pass,
        Span<const ImageView *>(&framebuf_refs.back(), 1),
        VkExtent2D { mem_req.width, mem_req.height },
        limits.maxFramebufferLayers);

      univ_offset += limits.maxFramebufferLayers;
    }

    // If there is a last framebuffer didn't use its full capacity, create it as
    // well.
    if (nuniv_last_framebuf != 0) {
      framebuf_refs.push_back(
        &evaluation_allocs.sim_univs_temps.emplace_back(
          do_img.view(univ_offset, nuniv_last_framebuf)));
      evaluation_allocs.sim_univs_temp_framebufs.emplace_back(
        ctxt, pipes.eval_pipe.pipe.pass,
        Span<const ImageView *>(&framebuf_refs.back(), 1),
        VkExtent2D { mem_req.width, mem_req.height },
        nuniv_last_framebuf);
    }
  }
  bool make() {
    if (!heap_mgr.make()) {
      return false;
    }
    for (auto& img_view : evaluation_allocs.sim_univs_temps) {
      if (!img_view.make()) {
        return false;
      }
    }
    for (auto& framebuf : evaluation_allocs.sim_univs_temp_framebufs) {
      if (!framebuf.make()) {
        return false;
      }
    }
    return true;
  }
  void drop() {
    for (auto& framebuf : evaluation_allocs.sim_univs_temp_framebufs) {
      framebuf.drop();
    }
    for (auto& img_view : evaluation_allocs.sim_univs_temps) {
      img_view.drop();
    }
    heap_mgr.drop();
  }
  ~CuvkAllocations() {
    drop();
  }
};


//
// CUVK Context.
//


struct Cuvk {
  Context ctxt;

  CuvkPipelines pipes;
  CuvkAllocations allocs;

  // We can't submit queues asynchronously.
  std::mutex submit_sync;
  
  Cuvk(const PhysicalDeviceInfo& phys_dev_info,
    const CuvkMemoryRequirements& mem_req) :
    ctxt(phys_dev_info, CUVK_PHYS_DEV_FEAT, CUVK_QUEUE_CAPS),
    pipes(ctxt, mem_req),
    allocs(ctxt, pipes, mem_req,
      MemoryAllocationGuidelines(ctxt, pipes, mem_req)) {}
  bool make() {
    return ctxt.make() && pipes.make() && allocs.make();
  }
  void drop() {
    allocs.drop();
    pipes.drop();
    ctxt.drop();
  }
  ~Cuvk() {
    drop();
  }
};

struct Task {
  const Cuvk& cuvk;

  Executable exec;
  DescriptorSet desc_set;
  Fence fence;

  std::future<CuvkTaskStatus> status;

  Task(const Cuvk& cuvk, const DescriptorSetLayout& desc_set_layout) :
    cuvk(cuvk),
    exec(cuvk.ctxt, cuvk.ctxt.queues[0]),
    desc_set(cuvk.ctxt, desc_set_layout),
    fence(cuvk.ctxt) {}
  bool make() {
    exec.make();
    desc_set.make();
    fence.make();
  }
};

std::string gen_phys_dev_json() {
  std::string rv;
  for (auto phys_dev_info : vk.phys_dev_infos) {
    // TODO: (penguinliong) Output should be JSON.
    rv += fmt::format("{} ({})\n",
      phys_dev_info.phys_dev_props.deviceName, 
      phys_dev_info.phys_dev_props.deviceType);
  }
  return std::move(rv);
}


CuvkResult L_STDCALL cuvkRedirectLog(const char* path) {
  // TODO: (penguinliong) Implement this.
  LOG.error("not implemented yet");
  return false;
}

CuvkResult L_STDCALL cuvkInitialize(CuvkBool debug) {
  if (!LOG.make()) {
    printf("failed to set up logger");
    std::terminate();
  }
  if (debug) {
    if (!vk.make_debug()) {
      return false;
    }
  } else {
    if (!vk.make()) {
      return false;
    }
  }
  phys_dev_json = gen_phys_dev_json();
  return true;
}

void L_STDCALL cuvkEnumeratePhysicalDevices(L_OUT char* pJson,
  L_INOUT CuvkSize* jsonSize) {
  if (*jsonSize) {
    *jsonSize = static_cast<CuvkSize>(phys_dev_json.size() + 1);
  } else {
    std::memcpy(pJson, phys_dev_json.c_str(), *jsonSize);
  }
}

L_EXPORT void L_STDCALL cuvkDeinitialize() {
  vk.drop();
  LOG.drop();
}

bool check_dev_cap(L_INOUT uint32_t& value, uint32_t limit,
  const std::string& value_desc) {
  if (limit < value) {
    LOG.warning("{} exceeds the limit of device (value={}; limit={})",
      value_desc, value, limit);
    value = limit;
    return false;
  }
  return true;
}
bool check_dev_caps(const VkPhysicalDeviceLimits& limits,
  L_INOUT CuvkMemoryRequirements& mem_req) {
  LOG.warning("as cuvk is still in progress, some variables can be constrained "
    "by hardware limits until workarounds are implemented");
  {
    auto limit = std::min({
      limits.maxComputeWorkGroupCount[0],
      limits.maxStorageBufferRange / (uint32_t)sizeof(DeformSpecs),
    });
    check_dev_cap(
      mem_req.nspec, limits.maxComputeWorkGroupCount[0],
      "(deformation) number of deform specs");
  } {
    auto limit = std::min({
      limits.maxComputeWorkGroupCount[1],
      limits.maxStorageBufferRange / (uint32_t)sizeof(Bacterium),
    });
    check_dev_cap(mem_req.nbac, limit, "(deformation) number of bacteria");
  } {
    auto limit = std::min({
      limits.maxComputeWorkGroupCount[0],
      limits.maxFramebufferLayers,
      limits.maxImageArrayLayers,
    });
    check_dev_cap(mem_req.nuniv, limit, "(evaluation) number of universes");
  } {
    auto limit = std::min({
      limits.maxComputeWorkGroupCount[1],
    });
    auto npack_univ = mem_req.width * mem_req.height / 4;
    auto nsec = npack_univ / limits.maxComputeWorkGroupSize[0];
    if (npack_univ % limits.maxComputeWorkGroupSize[0] != 0) {
      ++nsec;
    }
    if (!check_dev_cap(nsec, limit, "(evaluation) size of universes")) {
      return false;
    }
  }
  return true;
}

CuvkResult L_STDCALL cuvkCreateContext(
  CuvkSize physicalDeviceIndex,
  L_INOUT CuvkMemoryRequirements* memoryRequirements,
  L_OUT CuvkContext* pContext) {
  auto& phys_dev_info = vk.phys_dev_infos[physicalDeviceIndex];
  auto& limits = phys_dev_info.phys_dev_props.limits;
  // Ensure device is capable of the CUVK tasks.
  if (!check_dev_caps(limits, *memoryRequirements)) {
    return false;
  }
  // Create the context.
  auto rv = new Cuvk(phys_dev_info, *memoryRequirements);
  if (rv->make()) {
    (*pContext) = reinterpret_cast<CuvkContext>(rv);
    return true;
  } else {
    delete rv;
    return false;
  }
}

void L_STDCALL cuvkDestroyContext(
  CuvkContext context) {
  delete reinterpret_cast<Cuvk*>(context);
}



namespace deformation {
  using Invocation = CuvkDeformationInvocation;

  void write_desc_set(L_INOUT Task& task) {
    auto& allocs = task.cuvk.allocs.deformation_allocs;
    // Update descriptor set.
    task.desc_set
      .write(0, allocs.deform_specs, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
      .write(1, allocs.bacs, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
      .write(2, allocs.bacs_out, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
  }
  bool fill_cmd_buf(L_INOUT Task& task, const Invocation& invoke) {
    auto& allocs = task.cuvk.allocs.deformation_allocs;
    std::array<uint32_t, 3> meta {
      invoke.nBac,
      invoke.baseUniv,
      invoke.nUniv,
    };

    auto rec = task.exec.record();
    if (!rec.begin()) { return false; }
    rec
      // -----------------------------------------------------------------------
      // Wait for inputs to be fully written.
      .from_stage(VK_PIPELINE_STAGE_HOST_BIT)
        .barrier(allocs.deform_specs,
          VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
        .barrier(allocs.bacs,
          VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
      .to_stage(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)
      // -----------------------------------------------------------------------
      // Dispatch cell deformation.
      .push_const(task.cuvk.pipes.deform_pipe.pipe,
        0, static_cast<uint32_t>(meta.size() * sizeof(uint32_t)), meta.data())
      .dispatch(task.cuvk.pipes.deform_pipe.pipe, &task.desc_set,
        invoke.nSpec, invoke.nBac, 1)
      // -----------------------------------------------------------------------
      // Wait for host to read.
      .from_stage(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)
        .barrier(allocs.bacs_out,
          VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT)
      .to_stage(VK_PIPELINE_STAGE_HOST_BIT);
    return rec.end();
  }
  bool input(L_INOUT Task& task, const Invocation& invoke) {
    auto& allocs = task.cuvk.allocs.deformation_allocs;
    if (!allocs.deform_specs.dev_mem_view().send(
      invoke.pDeformSpecs, invoke.nSpec * sizeof(DeformSpecs))) {
      LOG.error("unable to send bacteria input");
      return false;
    }
    if (!allocs.bacs.dev_mem_view().send(
      invoke.pBacs, invoke.nBac * sizeof(Bacterium))) {
      LOG.error("unable to send deform specs input");
      return false;
    }
    return true;
  }
  bool output(L_INOUT Task& task, const Invocation& invoke) {
    auto& allocs = task.cuvk.allocs.deformation_allocs;
    if (!allocs.bacs_out.dev_mem_view().fetch(
      invoke.pBacsOut,
      invoke.nBac * invoke.nSpec * sizeof(Bacterium))) {
      LOG.error("unable to fetch bacteria output");
      return false;
    }
    return true;
  }
  CuvkTaskStatus worker_main(Cuvk* cuvk, L_INOUT Task* task,
    Invocation invoke) {
    write_desc_set(*task);
    // Prepare for execution.
    if (!fill_cmd_buf(*task, invoke)) {
      LOG.error("unable to fill command buffer for deformation task");
      return CUVK_TASK_STATUS_ERROR;
    }
    // Reset fence.
    if (!task->fence.make()) {
      return CUVK_TASK_STATUS_ERROR;
    }
    // Execute.
    { // std::scoped_lock _(ctxt->submit_sync)
      std::scoped_lock _(cuvk->submit_sync);
      // Send input.
      if (!input(*task, invoke)) {
        LOG.error("unable to send deformation input to device");
        return CUVK_TASK_STATUS_ERROR;
      }
      // Submit command buffer.
      if (!task->exec.execute().submit(task->fence)) {
        LOG.error("unable to submit deformation command buffer");
        return CUVK_TASK_STATUS_ERROR;
      }
      // TODO: (penguinliong) Remove this wait and move the output transfer to
      // polling.
      if (task->fence.wait() == FenceStatus::Error) {
        LOG.error("unable to wait the fence of deformation");
        return CUVK_TASK_STATUS_ERROR;
      }
      // Fetch output.
      if (!output(*task, invoke)) {
        LOG.error("unable to fetch deformation output from device");
        return CUVK_TASK_STATUS_ERROR;
      }
    } // std::scoped_lock _(ctxt->submit_sync)
    LOG.info("deformation task is done");
    return CUVK_TASK_STATUS_OK;
  }
  bool check_params(const Invocation& invoke) {
    // FIXME: (penguinliong) This check is not comprehensive.
    if (invoke.nSpec == 0) {
      LOG.warning("number of deform specs is 0; deform did nothing");
      return true;
    }
    if (invoke.nBac == 0) {
      LOG.warning("number of bacteria is 0; deform did nothing");
      return true;
    }
    if (invoke.nUniv == 0) {
      LOG.warning("number of universes is 0; deform did nothing");
      return true;
    }
    if (invoke.pDeformSpecs == nullptr) {
      LOG.error("`pDeformSpecs` is `nullptr`");
      return false;
    }
    if (invoke.pBacs == nullptr) {
      LOG.error("`pBacs` is `nullptr`");
      return false;
    }
    if (invoke.pBacsOut == nullptr) {
      LOG.error("`pBacsOut` is nullptr");
      return false;
    }
    return true;
  }
}
CuvkResult L_STDCALL cuvkInvokeDeformation(
  CuvkContext context,
  const CuvkDeformationInvocation* pInvocation,
  L_OUT CuvkTask* pTask) {
  auto invoke = *pInvocation;
  if (!deformation::check_params(invoke)) {
    return false;
  }

  // Create the task.
  auto cuvk = reinterpret_cast<Cuvk*>(context);
  auto task = new Task(*cuvk, cuvk->pipes.deform_pipe.pipe.desc_set_layout);
  if (!task->exec.make() || !task->desc_set.make()) {
    delete task;
    return false;
  }

  // Fill command buffer and execute asynchronously.
  task->status = std::async(deformation::worker_main, cuvk, task, invoke);
  LOG.info("dispatched deformation task");
  *pTask = reinterpret_cast<CuvkTask>(task);

  return true;
}



namespace evaluation {
  using Invocation = CuvkEvaluationInvocation;

  void write_desc_set(L_INOUT Task& task) {
    auto& allocs = task.cuvk.allocs.evaluation_allocs;
    task.desc_set
      .write(0, allocs.real_univ, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
      .write(1, allocs.sim_univs, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
      .write(2, allocs.sum_temp, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
      .write(3, allocs.partial_costs, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
  }
  bool fill_cmd_buf(L_INOUT Task& task, const Invocation& invoke) {
    using shader_interface::Bacterium;
    auto& allocs = task.cuvk.allocs.evaluation_allocs;

    auto rec = task.exec.record();
    if (!rec.begin()) { return false; }

    auto& limits = task.cuvk.ctxt.req.phys_dev_info->phys_dev_props.limits;
    auto bacs = reinterpret_cast<const Bacterium*>(invoke.pBacs);

    std::array<uint32_t, 1> eval_meta {
      invoke.baseUniv, // This will be added with `invoke` for multiple times.
    };

    rec
      // -----------------------------------------------------------------------
      // Wait for bacteria data to be written.
      .from_stage(VK_PIPELINE_STAGE_HOST_BIT)
        .barrier(allocs.bacs,
          VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT)
      .to_stage(VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);
    uint32_t bacs_pos = 0;
    uint32_t univ_pos = invoke.baseUniv;
    uint32_t bacs_offset = 0;
    uint32_t ngrp = (invoke.nSimUniv + limits.maxFramebufferLayers - 1)
      / limits.maxFramebufferLayers;
    for (auto i = 0u; i < ngrp; ++i) {
      bacs_pos = bacs_offset;
      // The number of universes that can be simulated is limited by the number
      // of layers that can be shoved into a single framebuffer.
      univ_pos += limits.maxFramebufferLayers;
      // Find the number of bacteria to be drawn in this batch. We assume that
      // all bacteria are sorted by universe ID.
      while (bacs_pos < invoke.nBac && bacs[bacs_pos].univ < univ_pos) {
        ++bacs_pos;
      }

      // Draw calls.
      auto nbac = bacs_pos - bacs_offset;
      auto& img_view = allocs.sim_univs_temps[i];
      auto& framebuf = allocs.sim_univs_temp_framebufs[i];
      rec
        // ---------------------------------------------------------------------
        // Rearrange simulated universes output layout.
        .from_stage(VK_PIPELINE_STAGE_VERTEX_INPUT_BIT)
          .barrier(img_view,
            0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
        .to_stage(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT)
        // ---------------------------------------------------------------------
        // Draw simulated cell universes.
        .push_const(task.cuvk.pipes.eval_pipe.pipe,
          VK_SHADER_STAGE_GEOMETRY_BIT,
          0, (uint32_t)eval_meta.size() * sizeof(uint32_t), eval_meta.data())
        .draw(task.cuvk.pipes.eval_pipe.pipe, {},
          allocs.bacs.slice(bacs_offset, nbac), nbac, framebuf)
        .copy_img_to_buf(allocs.sim_univs_temp_entire, allocs.sim_univs);

      // Update states.
      bacs_offset = bacs_pos;
      eval_meta[0] = univ_pos;
    }
    rec
      // -----------------------------------------------------------------------
      // Copy the simulated universes out.
      // TODO: (penguinliong) Use another queue for copy in the future.
      // -----------------------------------------------------------------------
      // Wait for the compute shader to read.
      .from_stage(VK_PIPELINE_STAGE_TRANSFER_BIT)
        .barrier(allocs.sim_univs,
          VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
      .to_stage(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    auto& scheduling = task.cuvk.pipes.cost_pipe.scheduling;

    if (scheduling.nsec != 0) {
      std::array<uint32_t, 4> cost_meta {
        scheduling.nsec_actual,
        scheduling.npack_sec,
        scheduling.npack_univ,
        0,
      };
      rec
        // ---------------------------------------------------------------------
        // Dispatch cost computation.
        .push_const(task.cuvk.pipes.cost_pipe.pipe_sec,
          0, (uint32_t)cost_meta.size() * sizeof(uint32_t), cost_meta.data())
        .dispatch(task.cuvk.pipes.cost_pipe.pipe_sec, &task.desc_set,
          invoke.nSimUniv, scheduling.nsec, 1);
    }
    if (scheduling.npack_res != 0) {
      std::array<uint32_t, 4> cost_meta {
        scheduling.nsec_actual,
        scheduling.npack_res,
        scheduling.npack_univ,
        scheduling.nsec,
      };
      rec
        // ---------------------------------------------------------------------
        // Dispatch cost computation for residuals.
        .push_const(task.cuvk.pipes.cost_pipe.pipe_res,
          0, (uint32_t)cost_meta.size() * sizeof(uint32_t), cost_meta.data())
        .dispatch(task.cuvk.pipes.cost_pipe.pipe_res, &task.desc_set,
          invoke.nSimUniv, 1, 1);
    }

    rec
      // -----------------------------------------------------------------------
      // Wait the costs to be computed and to be visible to host.
      .from_stage(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)
        .barrier(allocs.sim_univs,
          VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_HOST_READ_BIT)
        .barrier(allocs.partial_costs,
          VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT)
      .to_stage(VK_PIPELINE_STAGE_HOST_BIT);
    return rec.end();
  }
  bool input(L_INOUT Task& task, const Invocation& invoke) {
    auto& allocs = task.cuvk.allocs.evaluation_allocs;
    if (invoke.pBacs != nullptr) {
      if (!allocs.bacs.dev_mem_view().send(
        invoke.pBacs, invoke.nBac * sizeof(Bacterium))) {
        LOG.error("unable to send bacteria input");
        return false;
      }
    }
    if (invoke.pRealUniv != nullptr) {
      if (!allocs.real_univ.dev_mem_view().send(
        invoke.pRealUniv, invoke.width * invoke.height * sizeof(float))) {
        LOG.error("unable to send real universe input");
        return false;
      }
    }
    return true;
  }
  bool output(L_INOUT Task& task, const Invocation& invoke) {
    auto& allocs = task.cuvk.allocs.evaluation_allocs;
    auto& scheduling = task.cuvk.pipes.cost_pipe.scheduling;
    if (invoke.pSimUnivs == nullptr) {
      LOG.warning("the user application doesn't want the simulated universes");
    } else {
      if (!allocs.sim_univs.dev_mem_view().fetch(
        invoke.pSimUnivs,
        invoke.width * invoke.height * invoke.nSimUniv * sizeof(float))) {
        LOG.error("unable to fetch simulated universes");
        return false;
      }
    }
    if (invoke.pCosts == nullptr) {
      LOG.warning("the user application doesn't want the costs output");
    } else {
      auto nsec = scheduling.nsec_actual;
      auto size = invoke.nSimUniv * nsec * sizeof(float);
      auto mem = allocs.partial_costs.dev_mem_view().map(size);
      if (mem == nullptr) {
        LOG.error("unable to fetch costs output");
        return false;
      }
      auto data = reinterpret_cast<const float*>(mem);
      // Sum up the partial costs.
      // TODO: (penguinliong) Test if it will be faster we do this on GPU?
      auto univ = invoke.nSimUniv;
      auto costs = reinterpret_cast<float*>(invoke.pCosts);
      while (univ--) {
        auto univ_offset = univ * nsec;
        auto sec = nsec;
        costs[univ] = 0.;
        while (sec--) {
          costs[univ] += data[sec + univ_offset];
        }
      }
      allocs.partial_costs.dev_mem_view().unmap();
    }
    return true;
  }
  CuvkTaskStatus worker_main(Cuvk* cuvk, L_INOUT Task* task,
    Invocation invoke) {
    // Update descriptor set.
    evaluation::write_desc_set(*task);

    // Prepare for execution.
    if (!evaluation::fill_cmd_buf(*task, invoke)) {
      LOG.error("unable to fill command buffer for evaluation task");
      return CUVK_TASK_STATUS_ERROR;
    }
    if (!task->fence.make()) {
      return CUVK_TASK_STATUS_ERROR;
    }
    // Execute.
    { // std::scoped_lock _(ctxt->submit_sync)
      std::scoped_lock _(cuvk->submit_sync);

      // Send input.
      if (!input(*task, invoke)) {
        return CUVK_TASK_STATUS_ERROR;
      }
      if (!task->exec.execute().submit(task->fence)) {
        LOG.error("unable to submit command buffer");
        return CUVK_TASK_STATUS_ERROR;
      }
      // TODO: (penguinliong) Remove this wait and move the output transfer to
      // polling.
      if (task->fence.wait() == FenceStatus::Error) {
        LOG.error("unable to wait the fence");
        return CUVK_TASK_STATUS_ERROR;
      }
      // Fetch output.
      if (!output(*task, invoke)) {
        return CUVK_TASK_STATUS_ERROR;
      }
    } // std::scoped_lock _(ctxt->submit_sync)
    LOG.info("evaluation task is done");
    return CUVK_TASK_STATUS_OK;
  }
  bool check_params(const Invocation& invoke) {
    // FIXME: (penguinliong) This check is not comprehensive.
    if (invoke.nBac == 0) {
      LOG.warning("number of bacteria is 0; eval did nothing");
    }
    if (invoke.nSimUniv == 0) {
      LOG.warning("number of simulated universes is 0; eval did nothing");
    }
    if (invoke.width == 0 || invoke.height == 0) {
      LOG.warning("the size of universes to be drawn is 0, eval did nothing");
    }
    if (invoke.pSimUnivs == nullptr) {
      LOG.error("`pSimUnivs` is `nullptr`");
      return false;
    }
    if (invoke.pRealUniv == nullptr) {
      LOG.error("`pRealUniv` is `nullptr`");
      return false;
    }
    if (invoke.pBacs == nullptr) {
      LOG.error("`pBacs` is `nullptr`");
      return false;
    }
    if (invoke.pCosts == nullptr) {
      LOG.error("`pCosts` is `nullptr`");
      return false;
    }
    return true;
  }
}
CuvkResult L_STDCALL cuvkInvokeEvaluation(
  CuvkContext context,
  const CuvkEvaluationInvocation* pInvocation,
  L_OUT CuvkTask* pTask) {
  auto invoke = *pInvocation;
  if (!evaluation::check_params(invoke)) {
    return false;
  }

  // Create the task.
  auto cuvk = reinterpret_cast<Cuvk*>(context);
  auto task = new Task(*cuvk, cuvk->pipes.cost_pipe.pipe_sec.desc_set_layout);
  if (!task->exec.make() || !task->desc_set.make()) {
    delete task;
    return false;
  }

  // Fill command buffer and execute asynchronously.
  task->status = std::async(evaluation::worker_main, cuvk, task, invoke);
  LOG.info("dispatched evaluation task");
  *pTask = reinterpret_cast<CuvkTask>(task);

  return true;
}


CuvkTaskStatus L_STDCALL cuvkPoll(CuvkTask task) {
  auto& status = reinterpret_cast<Task*>(task)->status;
  try {
    if (status.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
      return status.get();
    } else {
      return CUVK_TASK_STATUS_NOT_READY;
    }
  } catch (const std::exception& e) {
    LOG.error("unexpected error occurred polling task: {}", e.what());
    return CUVK_TASK_STATUS_ERROR;
  }
}


void L_STDCALL cuvkDestroyTask(CuvkTask task) {
  delete reinterpret_cast<Task*>(task);
}
