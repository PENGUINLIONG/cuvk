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


std::vector<uint32_t> read_spirv(const std::string& path) {
  std::ifstream f(path, std::ios::in | std::ios::binary | std::ios::ate);
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


VkPhysicalDeviceFeatures cuvk_phys_dev_feat() {
  VkPhysicalDeviceFeatures feat {};
  feat.geometryShader = true;
  feat.shaderStorageBufferArrayDynamicIndexing = true;
  feat.shaderStorageImageArrayDynamicIndexing = true;
  return feat;
}


const std::array<VkDescriptorSetLayoutBinding, 3> deform_layout_binds {
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
};
const std::array<VkPushConstantRange, 1> deform_push_const_range {
  VkPushConstantRange { VK_SHADER_STAGE_COMPUTE_BIT, 0, 4 },
};

const std::array<VkDescriptorSetLayoutBinding, 0> eval_layout_binds {
};
const std::array<VkPushConstantRange, 1> eval_push_const_range {
  VkPushConstantRange { VK_SHADER_STAGE_GEOMETRY_BIT, 0, 4 },
};
const std::array<VkVertexInputBindingDescription, 1> eval_in_binds {
  VkVertexInputBindingDescription
  { 0, 6 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX }, // Bacterium
};
const std::array<VkVertexInputAttributeDescription, 4> eval_in_attrs {
  VkVertexInputAttributeDescription
  { 0, 0, VK_FORMAT_R32G32_SFLOAT, 0                    }, // pos
  { 1, 0, VK_FORMAT_R32G32_SFLOAT, 2 * sizeof(float)    }, // size
  { 2, 0, VK_FORMAT_R32_SFLOAT,    4 * sizeof(float)    }, // orient
  { 3, 0, VK_FORMAT_R32_UINT,      5 * sizeof(uint32_t) }, // univ
};
const std::array<VkPipelineColorBlendAttachmentState, 1> eval_out_blends {
  VkPipelineColorBlendAttachmentState
  { true,
    VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ONE, VK_BLEND_OP_MAX,
    VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ONE, VK_BLEND_OP_MAX,
    0xF,
  },
};
const std::array<VkAttachmentDescription, 1> eval_out_attach_descs {
  VkAttachmentDescription {
    0, VK_FORMAT_R32_SFLOAT, VK_SAMPLE_COUNT_1_BIT,
    VK_ATTACHMENT_LOAD_OP_CLEAR,
    VK_ATTACHMENT_STORE_OP_STORE,
    VK_ATTACHMENT_LOAD_OP_DONT_CARE,
    VK_ATTACHMENT_STORE_OP_DONT_CARE,
    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    VK_IMAGE_LAYOUT_GENERAL,
  },
};
const std::array<VkAttachmentReference, 1> eval_out_attach_refs {
  VkAttachmentReference
  { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL },
};

const std::array<VkDescriptorSetLayoutBinding, 4> cost_layout_binds {
  VkDescriptorSetLayoutBinding
  // Bacterium[] bacs
  { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
    1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
  // image2D real_univ
  { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
    1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
  // image2DArray real_univ
  { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
  // float[] costs
  { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
};
const std::array<VkPushConstantRange, 1> cost_push_const_rng{
  VkPushConstantRange { VK_SHADER_STAGE_COMPUTE_BIT, 0, 12 },
};

struct DeformAllocation {
  BufferSlice deform_specs;
  BufferSlice bacs;
  BufferSlice bacs_out;
};
struct EvalAllocation {
  BufferSlice bacs;
  ImageView sim_univs;
};
struct CostAllocation {
  ImageView real_univ;
  BufferSlice temp;
  BufferSlice costs;
};
struct StagingAllocation {
  BufferSlice real_univ_staging;
  BufferSlice sim_univs_staging;
};


template<typename TElem>
constexpr VkDeviceSize inflated(uint32_t n) {
  return n * sizeof(TElem);
}

constexpr VkDeviceSize align(VkDeviceSize size, VkDeviceSize alignment) {
  return (size + alignment - 1) / alignment * alignment;
}
constexpr VkDeviceSize offset_storage_buf(
  const PhysicalDeviceInfo& phys_dev_info, VkDeviceSize size) {
  return align(size, phys_dev_info.phys_dev_props
    .limits.minStorageBufferOffsetAlignment);
}
constexpr VkDeviceSize align_map_mem(
  const PhysicalDeviceInfo& phys_dev_info, VkDeviceSize size) {
  return align(size, phys_dev_info.phys_dev_props
    .limits.minMemoryMapAlignment);
}

struct MemoryAllocationGuidelines {
  struct DeformAllocationGuideline {
    VkDeviceSize deform_specs_offset;
    VkDeviceSize deform_specs_size;
    VkDeviceSize bacs_offset;
    VkDeviceSize bacs_size;
    VkDeviceSize bacs_out_offset;
    VkDeviceSize bacs_out_size;
  } deform;
  struct EvalAllocationGuideline {
    VkDeviceSize bacs_offset;
    VkDeviceSize bacs_size;
    uint32_t sim_univs_base_layer;
    uint32_t sim_univs_nlayer;
  } eval;
  struct CostAllocationGuideline {
    uint32_t real_univ_base_layer;
    uint32_t real_univ_nlayer;
    VkDeviceSize temp_offset;
    VkDeviceSize temp_size;
    VkDeviceSize costs_offset;
    VkDeviceSize costs_size;
  } cost;
  struct StagingAllocationGuideline {
    VkDeviceSize real_univ_staging_offset;
    VkDeviceSize real_univ_staging_size;
    VkDeviceSize sim_univs_staging_offset;
    VkDeviceSize sim_univs_staging_size;
  } staging;
  struct MetadataAllocationGuideline {
    VkDeviceSize meta_offset;
    VkDeviceSize meta_size;
  };

  VkDeviceSize host_visible_buf_size;
  uint32_t host_visible_img_nlayer;
  VkDeviceSize device_only_buf_size;
  uint32_t device_only_img_nlayer;

  MemoryAllocationGuidelines(
    const PhysicalDeviceInfo& phys_dev_info, const CuvkMemoryRequirements mem_req) {

    // Host-visible buffer.

    deform.deform_specs_offset = 0;
    deform.deform_specs_size = inflated<DeformSpecs>(mem_req.nspec);

    deform.bacs_offset =
      offset_storage_buf(phys_dev_info,
        deform.deform_specs_offset + deform.deform_specs_size);
    deform.bacs_size = inflated<Bacterium>(mem_req.nbac);

    deform.bacs_out_offset =
      offset_storage_buf(phys_dev_info, deform.bacs_offset + deform.bacs_size);
    deform.bacs_out_size = inflated<Bacterium>(mem_req.nspec * mem_req.nbac);

    if (mem_req.shareBacteriaBuffer) {
      eval.bacs_offset = deform.bacs_out_offset;
    } else {
      eval.bacs_offset =
        offset_storage_buf(phys_dev_info,
          deform.bacs_out_offset + deform.bacs_out_size);
    }
    eval.bacs_size = inflated<Bacterium>(mem_req.nspec * mem_req.nbac);

    cost.costs_offset =
      offset_storage_buf(phys_dev_info, eval.bacs_offset + eval.bacs_size);
    cost.costs_size = inflated<float>(mem_req.nuniv);

    // Note that we are not duplexing. Real universes are always inputs,
    // simulated universes are always outputs, and cost computation is a part of
    // the execution of an evaluation task. We don't need to consider the impact
    // of memory sharing.
    staging.real_univ_staging_offset =
      align_map_mem(phys_dev_info, cost.costs_offset + cost.costs_size);
    staging.real_univ_staging_size =
      inflated<float>(mem_req.width * mem_req.height);
    staging.sim_univs_staging_offset = staging.real_univ_staging_offset;
    staging.sim_univs_staging_size =
      inflated<float>(mem_req.width * mem_req.height * mem_req.nuniv);

    // Device-only buffer.

    cost.temp_offset = 0;
    cost.temp_size =
      inflated<float>(mem_req.width * mem_req.height * mem_req.nuniv / 4);

    // Device-only image.

    cost.real_univ_base_layer = 0;
    cost.real_univ_nlayer = 1;

    eval.sim_univs_base_layer =
      cost.real_univ_base_layer + cost.real_univ_nlayer;
    eval.sim_univs_nlayer = mem_req.nuniv;

    // Summary.
    host_visible_buf_size = align_map_mem(phys_dev_info,
      staging.sim_univs_staging_offset + staging.sim_univs_staging_size);
    device_only_buf_size = cost.temp_offset + cost.temp_size;
    device_only_img_nlayer = eval.sim_univs_base_layer + eval.sim_univs_nlayer;
  }
};

std::array<VkQueueFlags, 1> CUVK_QUEUE_CAPS = {
  VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT,
};
VkPhysicalDeviceFeatures CUVK_PHYS_DEV_FEAT = cuvk_phys_dev_feat();

struct Cuvk {

  Context ctxt;
  // We can't submit queues asynchronously.
  std::mutex submit_sync;


  ShaderManager shader_mgr;
  PipelineManager pipe_mgr;
  HeapManager heap_mgr;


  const Shader* deform_comp_shader;
  const Shader* eval_vert_shader;
  const Shader* eval_geom_shader;
  const Shader* eval_frag_shader;
  const Shader* cost_comp_shader;

  ShaderStage deform_stage;
  std::array<ShaderStage, 3> eval_stages;
  ShaderStage cost_stage;


  const BufferAllocation* host_visible_buf;
  const BufferAllocation* device_only_buf;
  const ImageAllocation* device_only_img;

  DeformAllocation deform_alloc;
  EvalAllocation eval_alloc;
  CostAllocation cost_alloc;
  StagingAllocation staging_alloc;


  const ComputePipeline* deform_pipe;
  const GraphicsPipeline* eval_pipe;
  const ComputePipeline* cost_pipe;

  

  Cuvk(const PhysicalDeviceInfo& phys_dev_info,
    const CuvkMemoryRequirements& mem_req,
    const MemoryAllocationGuidelines& guide) :
    ctxt(phys_dev_info, CUVK_PHYS_DEV_FEAT, CUVK_QUEUE_CAPS),

    // Compile shaders.
    shader_mgr(ctxt),

    deform_comp_shader(&shader_mgr.declare_shader(
      read_spirv("assets/shaders/deform.comp.spv"))),
    eval_vert_shader(&shader_mgr.declare_shader(
      read_spirv("assets/shaders/eval.vert.spv"))),
    eval_geom_shader(&shader_mgr.declare_shader(
      read_spirv("assets/shaders/eval.geom.spv"))),
    eval_frag_shader(&shader_mgr.declare_shader(
      read_spirv("assets/shaders/eval.frag.spv"))),
    cost_comp_shader(&shader_mgr.declare_shader(
      read_spirv("assets/shaders/cost.comp.spv"))),

    // Construct pipelines.
    pipe_mgr(ctxt),

    deform_stage(
      deform_comp_shader->stage("main", VK_SHADER_STAGE_COMPUTE_BIT)
    ),
    eval_stages({
      eval_vert_shader->stage("main", VK_SHADER_STAGE_VERTEX_BIT),
      eval_geom_shader->stage("main", VK_SHADER_STAGE_GEOMETRY_BIT),
      eval_frag_shader->stage("main", VK_SHADER_STAGE_FRAGMENT_BIT),
    }),
    cost_stage(
      cost_comp_shader->stage("main", VK_SHADER_STAGE_COMPUTE_BIT)
    ),


    // Allocate memory and create memory-dependent objects.
    heap_mgr(ctxt),

    host_visible_buf(&heap_mgr.declare_buf(
      guide.host_visible_buf_size,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      MemoryVisibility::HostVisible)),
    device_only_buf(&heap_mgr.declare_buf(
      guide.device_only_buf_size,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      MemoryVisibility::DeviceOnly)),
    device_only_img(&heap_mgr.declare_img(
      { mem_req.width, mem_req.height },
      guide.device_only_img_nlayer,
      VK_FORMAT_R32_SFLOAT,
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_STORAGE_BIT |
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
      VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      VK_IMAGE_TILING_OPTIMAL,
      MemoryVisibility::DeviceOnly)),

    deform_alloc({
      host_visible_buf->slice(
        guide.deform.deform_specs_offset, guide.deform.deform_specs_size),
      host_visible_buf->slice(
        guide.deform.bacs_offset, guide.deform.bacs_size),
      host_visible_buf->slice(
        guide.deform.bacs_out_offset, guide.deform.bacs_out_size),
    }),
    eval_alloc({
      host_visible_buf->slice(
        guide.eval.bacs_offset, guide.eval.bacs_size),
      device_only_img->view(
        guide.eval.sim_univs_base_layer, guide.eval.sim_univs_nlayer),
    }),
    cost_alloc({
      device_only_img->view(
        guide.cost.real_univ_base_layer, {}),
      device_only_buf->slice(
        guide.cost.temp_offset, guide.cost.temp_size),
      host_visible_buf->slice(
        guide.cost.costs_offset, guide.cost.costs_size),
    }),
    staging_alloc({
      host_visible_buf->slice(
        guide.staging.real_univ_staging_offset,
        guide.staging.real_univ_staging_size),
      host_visible_buf->slice(
        guide.staging.sim_univs_staging_offset,
        guide.staging.sim_univs_staging_size),
    }),

    deform_pipe(&pipe_mgr.declare_comp_pipe("deform", &deform_stage,
      deform_push_const_range, deform_layout_binds)),
    eval_pipe(&pipe_mgr.declare_graph_pipe("eval", eval_stages,
      eval_push_const_range, eval_layout_binds,
      eval_in_binds, eval_in_attrs, eval_out_blends,
      eval_out_attach_descs, eval_out_attach_refs, eval_alloc.sim_univs)),
    cost_pipe(&pipe_mgr.declare_comp_pipe("cost", &cost_stage,
      cost_push_const_rng, cost_layout_binds)) {}

  bool make() {
    return ctxt.make() &&
      shader_mgr.make(false) && heap_mgr.make() &&
      eval_alloc.sim_univs.make() && cost_alloc.real_univ.make() &&
      pipe_mgr.make();
  }
  void drop() {
    pipe_mgr.drop();
    eval_alloc.sim_univs.drop();
    cost_alloc.real_univ.drop();
    heap_mgr.drop();
    shader_mgr.drop();
    ctxt.drop();
  }
  ~Cuvk() { drop(); }
};


struct Task {
  const Cuvk& ctxt;

  Executable exec;
  DescriptorSet desc_set;
  Fence fence;

  std::future<CuvkTaskStatus> status;
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
    std::memcpy(pJson, phys_dev_json.data(), *jsonSize);
  }
}

CuvkResult L_STDCALL cuvkCreateContext(
  CuvkSize physicalDeviceIndex,
  L_INOUT CuvkMemoryRequirements memoryRequirements,
  L_OUT CuvkContext* pContext) {
  auto& phys_dev_info = vk.phys_dev_infos[physicalDeviceIndex];
  auto& limits = phys_dev_info.phys_dev_props.limits;
  if (limits.maxImageArrayLayers < memoryRequirements.nuniv) {
    LOG.info("number of universes rendered in one eval task exceeds the limit "
      "of device (nuniv={}; limit={})",
      memoryRequirements.nuniv, limits.maxImageArrayLayers);
    memoryRequirements.nuniv = limits.maxImageArrayLayers;
  }
  if (limits.maxComputeWorkGroupCount[0] < memoryRequirements.nspec) {
    LOG.info("number of deform specs used in deform task exceeds the limit of "
      "device (nspec={}; limit={})",
      memoryRequirements.nspec, limits.maxComputeWorkGroupCount[0]);
    memoryRequirements.nspec = limits.maxComputeWorkGroupCount[0];
  }
  if (limits.maxComputeWorkGroupCount[1] < memoryRequirements.nbac) {
    LOG.info("number of bacteria used in deform task exceeds the limit of "
      "device (nbac={}; limit={})",
      memoryRequirements.nbac, limits.maxComputeWorkGroupCount[1]);
    memoryRequirements.nbac = limits.maxComputeWorkGroupCount[1];
  }
  if (limits.maxComputeWorkGroupCount[0] < memoryRequirements.width) {
    LOG.error("width of universe exceeds the limit of device (width={}, "
      "limit={})",
      memoryRequirements.height, limits.maxComputeWorkGroupCount[0]);
    return false;
  }
  if (limits.maxComputeWorkGroupCount[1] < memoryRequirements.height) {
    LOG.error("height of universe exceeds the limit of device (height={}, "
      "limit={})",
      memoryRequirements.height, limits.maxComputeWorkGroupCount[1]);
    return false;
  }
  auto rv = new Cuvk(
    phys_dev_info, memoryRequirements,
    MemoryAllocationGuidelines(phys_dev_info, memoryRequirements));
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

bool fill_deform(Task& task,
  BufferSlice deform_specs, uint32_t nspec,
  BufferSlice bacs, uint32_t nbac,
  BufferSlice bacs_out) {

  auto rec = task.exec.record();
  if (!rec.begin()) { return false; }

  rec
    // -------------------------------------------------------------------------
    // Wait for inputs to be fully written.
    .from_stage(VK_PIPELINE_STAGE_HOST_BIT)
      .barrier(deform_specs,
        VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
      .barrier(bacs,
        VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT)
      .barrier(bacs_out,
        0, VK_ACCESS_SHADER_WRITE_BIT)
    .to_stage(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)
    // -------------------------------------------------------------------------
    // Dispatch cell deformation.
    .push_const(*task.ctxt.deform_pipe,
      0, sizeof(uint32_t), &nspec)
    .dispatch(*task.ctxt.deform_pipe, &task.desc_set, nspec, nbac, 1)
    // -------------------------------------------------------------------------
    // Wait for host to read.
    .from_stage(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)
      .barrier(bacs_out,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT)
    .to_stage(VK_PIPELINE_STAGE_HOST_BIT);

  return rec.end();
}
CuvkResult L_STDCALL cuvkInvokeDeformation(
  CuvkContext context,
  const CuvkDeformationInvocation* pInvocation,
  L_OUT CuvkTask* pTask) {
  auto invoke = *pInvocation;
  if (invoke.nSpec == 0) {
    LOG.warning("number of deform specs is 0; deform did nothing");
    return true;
  }
  if (invoke.nBac == 0) {
    LOG.warning("number of bacteria is 0; deform did nothing");
    return true;
  }

  // Create the task.
  auto ctxt = reinterpret_cast<Cuvk*>(context);
  auto task = new Task {
    *ctxt,
    { ctxt->ctxt, ctxt->ctxt.queues[0] },
    { ctxt->ctxt, *ctxt->deform_pipe },
    { ctxt->ctxt }, {},
  };
  if (!task->exec.make() || !task->desc_set.make()) {
    delete task;
    return false;
  }

  // Fill command buffer and execute asynchronously.
  task->status = std::async([=]{
    // Update descriptor set.
    task->desc_set
      .write(0, ctxt->deform_alloc.deform_specs,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
      .write(1, ctxt->deform_alloc.bacs,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
      .write(2, ctxt->deform_alloc.bacs_out,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    // Prepare for execution.
    if (!fill_deform(*task,
      ctxt->deform_alloc.deform_specs, invoke.nSpec,
      ctxt->deform_alloc.bacs, invoke.nBac,
      ctxt->deform_alloc.bacs_out)) {
      LOG.error("unable to fill command buffer with deform task commands");
      return CUVK_TASK_STATUS_ERROR;
    }
    if (!task->fence.make()) {
      return CUVK_TASK_STATUS_ERROR;
    }
    // Execute.
    { // std::scoped_lock _(ctxt->submit_sync)
      std::scoped_lock _(ctxt->submit_sync);
      // Send inputs to device.
      if (invoke.pDeformSpecs != nullptr) {
        if (!ctxt->deform_alloc.deform_specs.dev_mem_view().send(
          invoke.pDeformSpecs, invoke.nSpec * sizeof(DeformSpecs))) {
          LOG.error("unable to send bacteria input");
          return CUVK_TASK_STATUS_ERROR;
        }
      }
      if (invoke.pBacs != nullptr) {
        if (!ctxt->deform_alloc.bacs.dev_mem_view().send(
          invoke.pBacs, invoke.nBac * sizeof(Bacterium))) {
          LOG.error("unable to send deform specs input");
          return CUVK_TASK_STATUS_ERROR;
        }
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
      // Fetch back output.
      if (invoke.pBacsOut == nullptr) {
        if (ctxt->deform_alloc.bacs_out.offset != ctxt->eval_alloc.bacs.offset) {
          LOG.warning("deform output is not shared with eval input, and the user "
            "application don't want to fetch back results");
        }
      } else {
        if (!ctxt->deform_alloc.bacs_out.dev_mem_view().fetch(
          invoke.pBacsOut,
          invoke.nBac * invoke.nSpec * sizeof(Bacterium))) {
          LOG.error("unable to fetch bacteria output");
          return CUVK_TASK_STATUS_ERROR;
        }
      }
    } // std::scoped_lock _(ctxt->submit_sync)
    LOG.info("deform task is done");
    return CUVK_TASK_STATUS_OK;
  });
  LOG.info("dispatched deform task");
  *pTask = reinterpret_cast<CuvkTask>(task);

  return true;
}

bool fill_eval_workset(Task& task,
  BufferSlice bacs, uint32_t nbac,
  const ImageView& sim_univs, uint32_t base_univ, uint32_t nuniv,
  const ImageView& real_univ,
  BufferSlice real_univ_staging, BufferSlice sim_univs_staging,
  BufferSlice costs) {

  auto extent = real_univ.img_slice.img_alloc->req.extent;
  auto half_w = extent.width / 2;
  auto half_h = extent.height / 2;

  auto rec = task.exec.record();
  if (!rec.begin()) { return false; }

  std::array<uint32_t, 3> meta = { half_w, half_h, half_w * half_h };

  rec
    // -------------------------------------------------------------------------
    // Wait for bacteria data to be written.
    .from_stage(VK_PIPELINE_STAGE_HOST_BIT)
      .barrier(bacs,
        VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT)
    .to_stage(VK_PIPELINE_STAGE_VERTEX_INPUT_BIT)
    // -------------------------------------------------------------------------
    // Wait for the real universe to be loaded in staging buffer. Re-layout the
    // device-accessible real universe.
    .from_stage(VK_PIPELINE_STAGE_HOST_BIT)
      .barrier(real_univ_staging,
        VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT)
      .barrier(real_univ,
        0, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
    .to_stage(VK_PIPELINE_STAGE_TRANSFER_BIT)
    // -------------------------------------------------------------------------
    // Stage the real universe.
    .copy_buf_to_img(real_univ_staging, real_univ)
    // -------------------------------------------------------------------------
    // We are sharing staging buffer for the real universe and the simulated
    // universes. Wait for idle before we use it again.
    .from_stage(VK_PIPELINE_STAGE_TRANSFER_BIT)
      .barrier(real_univ_staging,
        VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT)
    .to_stage(VK_PIPELINE_STAGE_TRANSFER_BIT)
    // -------------------------------------------------------------------------
    // Wait for real universe to be ready.
    .from_stage(VK_PIPELINE_STAGE_TRANSFER_BIT)
      .barrier(real_univ,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL)
    .to_stage(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)
    // -------------------------------------------------------------------------
    // Rearrange simulated universes output layout.
    .from_stage(VK_PIPELINE_STAGE_VERTEX_INPUT_BIT)
      .barrier(sim_univs,
        0, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
    .to_stage(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT)
    // -------------------------------------------------------------------------
    // Draw simulated cell universes.
    .push_const(*task.ctxt.eval_pipe, VK_SHADER_STAGE_GEOMETRY_BIT,
      0, sizeof(uint32_t), &base_univ)
    .draw(*task.ctxt.eval_pipe, {}, bacs, nbac)
    // -------------------------------------------------------------------------
    // Sync to transfer output simulations to staging buffer.
    .from_stage(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT)
      .barrier(sim_univs,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
    .to_stage(VK_PIPELINE_STAGE_TRANSFER_BIT)
    // -------------------------------------------------------------------------
    // Copy the simulated universes out.
    .copy_img_to_buf(sim_univs, sim_univs_staging)
    // -------------------------------------------------------------------------
    // Wait for the host to read.
    .from_stage(VK_PIPELINE_STAGE_TRANSFER_BIT)
      .barrier(sim_univs_staging,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT)
    .to_stage(VK_PIPELINE_STAGE_HOST_BIT)
    .from_stage(VK_PIPELINE_STAGE_TRANSFER_BIT)
      .barrier(sim_univs,
        VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL)
    .to_stage(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)
    // -------------------------------------------------------------------------
    // Dispatch cost computation.
    .push_const(*task.ctxt.cost_pipe,
      0, static_cast<uint32_t>(meta.size() * sizeof(uint32_t)), meta.data())
    .dispatch(*task.ctxt.cost_pipe, &task.desc_set, half_w, half_h, nuniv)
    // -------------------------------------------------------------------------
    // Wait the costs to be computed and to be visible to host.
    .from_stage(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)
      .barrier(costs,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT)
    .to_stage(VK_PIPELINE_STAGE_HOST_BIT);
  return rec.end();
}
CuvkResult L_STDCALL cuvkInvokeEvaluation(
  CuvkContext context,
  const CuvkEvaluationInvocation* pInvocation,
  L_OUT CuvkTask* pTask) {
  auto invoke = *pInvocation;
  if (invoke.nBac == 0) {
    LOG.warning("number of bacteria is 0; eval did nothing");
    return true;
  }
  if (invoke.nSimUniv == 0) {
    LOG.warning("number of simulated universes is 0; eval did nothing");
    return true;
  }


  // Create the task.
  auto ctxt = reinterpret_cast<Cuvk*>(context);
  auto extent = ctxt->eval_pipe->attach->img_slice.img_alloc->req.extent;
  auto task = new Task {
    *ctxt,
    { ctxt->ctxt, ctxt->ctxt.queues[0] },
    { ctxt->ctxt, *ctxt->cost_pipe },
    { ctxt->ctxt }, {},
  };
  if (!task->exec.make() || !task->desc_set.make()) {
    delete task;
    return false;
  }

  // Fill command buffer and execute asynchronously.
  task->status = std::async([=]{
    // Update descriptor set.
    task->desc_set
      .write(0, ctxt->cost_alloc.real_univ,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
      .write(1, ctxt->eval_alloc.sim_univs,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
      .write(2, ctxt->cost_alloc.temp,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
      .write(3, ctxt->cost_alloc.costs,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    // Prepare for execution.
    if (!fill_eval_workset(*task,
      task->ctxt.eval_alloc.bacs, invoke.nBac,
      task->ctxt.eval_alloc.sim_univs, invoke.baseUniv, invoke.nSimUniv,
      task->ctxt.cost_alloc.real_univ,
      task->ctxt.staging_alloc.real_univ_staging,
      task->ctxt.staging_alloc.sim_univs_staging,
      task->ctxt.cost_alloc.costs)) {
      return CUVK_TASK_STATUS_ERROR;
    }
    if (!task->fence.make()) {
      return CUVK_TASK_STATUS_ERROR;
    }
    // Execute.
    { // std::scoped_lock _(ctxt->submit_sync)
      std::scoped_lock _(ctxt->submit_sync);

      // Send inputs to device.
      if (invoke.pBacs != nullptr) {
        if (!ctxt->deform_alloc.bacs.dev_mem_view().send(//////////////
          invoke.pBacs, invoke.nBac * sizeof(Bacterium))) {
          LOG.error("unable to send bacteria input");
          return CUVK_TASK_STATUS_ERROR;
        }
      }
      if (invoke.pRealUniv != nullptr) {
        if (!ctxt->staging_alloc.real_univ_staging.dev_mem_view().send(
          invoke.pRealUniv, invoke.realUnivSize)) {
          LOG.error("unable to send real universe input");
          return CUVK_TASK_STATUS_ERROR;
        }
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
      // Fetch back output.
      if (invoke.pSimUnivs == nullptr) {
        LOG.warning("the user application doesn't want the simulated universes");
      } else {
        if (!ctxt->staging_alloc.sim_univs_staging.dev_mem_view().fetch(
          invoke.pSimUnivs,
          extent.width * extent.height * invoke.nSimUniv * sizeof(float))) {
          LOG.error("unable to fetch simulated universes");
          return CUVK_TASK_STATUS_ERROR;
        }
      }
      if (invoke.pCosts == nullptr) {
        LOG.warning("the user application doesn't want the costs output");
      } else {
        if (!ctxt->cost_alloc.costs.dev_mem_view().fetch(
          invoke.pCosts, invoke.nSimUniv * sizeof(float))) {
          LOG.error("unable to fetch costs output");
          return CUVK_TASK_STATUS_ERROR;
        }
      }
    } // std::scoped_lock _(ctxt->submit_sync)
    LOG.info("eval task is done");
    return CUVK_TASK_STATUS_OK;
  });
  LOG.info("dispatched eval task");
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
