#include <cuvk/context.hpp>
#include <cuvk/workflow.hpp>
#include <cuvk/logger.hpp>
#include <cuvk/shader_interface.hpp>
#include <cstdlib>

using namespace cuvk;
using namespace cuvk::shader_interface;


void select_phys_dev(Context& ctxt) {
  auto infos = ctxt.enum_phys_dev();
  int idx = -1;
  while (idx < 0 || idx >= infos.size()) {
    idx = 0;
    //std::scanf("%d", &idx);
  }
  ctxt.select_phys_dev(infos[idx]);
}

int main() {
  auto ctxt = std::make_shared<Context>();
  select_phys_dev(*ctxt);

  //
  // Deformation
  //

  auto deform = ctxt->make_contextual<Deformation>();

  StorageMeasure deform_in_size = [ctxt]{ return 
    ctxt->get_aligned_size(sizeof(DeformSpecs), BufferType::StorageBuffer) +
    ctxt->get_aligned_size(sizeof(Bacterium), BufferType::StorageBuffer);
  };
  StorageMeasure deform_out_size = [ctxt]{ return
    ctxt->get_aligned_size(sizeof(Bacterium), BufferType::StorageBuffer);
  };

  StorageMeasure specs_aligned = [ctxt]{ return
    ctxt->get_aligned_size(sizeof(DeformSpecs), BufferType::StorageBuffer);
  };

  auto deform_in_buf = ctxt->make_contextual<StorageBuffer>(
    deform_in_size,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  auto deform_out_buf = ctxt->make_contextual<StorageBuffer>(
    deform_out_size,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);


  StorageMeasure deform_in_alloc_size =
    [deform_in_buf]{ return deform_in_buf->alloc_size(); };
  StorageMeasure deform_out_alloc_size =
    [deform_out_buf]{ return deform_out_buf->alloc_size(); };

  auto deform_in_mem = ctxt->make_contextual<StagingStorage>(
    deform_in_alloc_size,
    StorageOptimization::Send);
  auto deform_out_mem = ctxt->make_contextual<StagingStorage>(
    deform_out_alloc_size,
    StorageOptimization::Fetch);

  deform_in_buf->bind(*deform_in_mem);
  deform_out_buf->bind(*deform_out_mem);

  // Set up input data.
  DeformSpecs spec;
  spec.rotate = 1.;
  spec.stretch = { 1., 1. };
  spec.translate = { 1., 1. };
  deform_in_mem->send(&spec, 0, sizeof(DeformSpecs));

  Bacterium bac;
  bac.orient = 1.;
  bac.pos = { 2., 3. };
  bac.size = { 4., 5. };
  deform_in_mem->send(&bac, specs_aligned, sizeof(Bacterium));

  // Dispatch deformation.
  deform->execute(
    deform_in_buf->view(0, sizeof(DeformSpecs)),
    deform_in_buf->view(specs_aligned, sizeof(Bacterium)),
    deform_out_buf->view());

  Bacterium bac_out;
  //deform_out_mem->fetch(&bac_out, 0, sizeof(Bacterium));
  deform_out_mem->fetch(&bac_out, 0, sizeof(Bacterium));

  LOG.info("Deformed bactrium: orient={}, pos=({}, {}), size=({}, {})",
    bac_out.orient,
    bac_out.pos[0], bac_out.pos[1],
    bac_out.size[0], bac_out.size[1]);



  //
  // Evaluation
  //

  auto eval = ctxt->make_contextual<Evaluation>(360, 240, 1);

  auto real_univ = ctxt->make_contextual<UniformStorageImage>(
    VkExtent2D{ 360, 240 }, std::nullopt, VK_FORMAT_R32_SFLOAT);
  auto real_univ_mem = ctxt->make_contextual<StagingStorage>(
    (StorageMeasure)[=]{ return real_univ->alloc_size(); },
    StorageOptimization::Send);
  real_univ->bind(*real_univ_mem, 0);
  
  auto real_univ_buf = ctxt->make_contextual<StorageBuffer>(
    (StorageMeasure)[=]{ return real_univ->size(); },
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  auto real_univ_buf_mem = ctxt->make_contextual<StagingStorage>(
    (StorageMeasure)[=]{ return real_univ_buf->alloc_size(); },
    StorageOptimization::Send);
  real_univ_buf->bind(*real_univ_buf_mem, 0);


  auto sim_univs = ctxt->make_contextual<ColorAttachmentStorageImage>(
    VkExtent2D{ 360, 240 }, 1, VK_FORMAT_R32_SFLOAT);
  auto sim_univs_mem = ctxt->make_contextual<DeviceOnlyStorage>(
    (StorageMeasure)[=]{ return sim_univs->alloc_size(); });
  sim_univs->bind(*sim_univs_mem, 0);

  auto sim_univs_buf = ctxt->make_contextual<StorageBuffer>(
    (StorageMeasure)[=]{ return sim_univs->size(); },
    VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  auto sim_univs_buf_mem = ctxt->make_contextual<StagingStorage>(
    (StorageMeasure)[=]{ return sim_univs_buf->alloc_size(); },
    StorageOptimization::Fetch);
  sim_univs_buf->bind(*sim_univs_buf_mem, 0);

  auto cost_buf = ctxt->make_contextual<StorageBuffer>(
    (StorageMeasure)[=]{ return sizeof(float); },
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  auto cost_buf_mem = ctxt->make_contextual<StagingStorage>(
    (StorageMeasure)[=]{ return cost_buf->alloc_size(); },
    StorageOptimization::Fetch);
  cost_buf->bind(*cost_buf_mem, 0);

  std::vector<char> buf(sim_univs_buf->size(), 0);

  real_univ_buf_mem->send(buf.data(), 0, buf.size());

  eval->execute(deform_in_buf->view(),
    real_univ_buf->view(),
    real_univ->view(),
    sim_univs_buf->view(),
    sim_univs->view(),
    cost_buf->view());

  cost_buf_mem->fetch(buf.data(), 0, cost_buf->size());

  std::getc(stdin);
  return 0;
}
