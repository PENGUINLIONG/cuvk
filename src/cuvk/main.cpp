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
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

  auto deform_in_mem = ctxt->make_contextual<StagingStorage>(
    StorageOptimization::Send);
  auto deform_out_mem = ctxt->make_contextual<StagingStorage>(
    StorageOptimization::Fetch);

  deform_in_buf->bind(*deform_in_mem);
  deform_out_buf->bind(*deform_out_mem);



  //
  // Evaluation
  //

  auto eval = ctxt->make_contextual<Evaluation>(360, 240, 1);

  auto real_univ = ctxt->make_contextual<UniformStorageImage>(
    VkExtent2D{ 360, 240 }, std::nullopt, VK_FORMAT_R32_SFLOAT);
  auto real_univ_mem = ctxt->make_contextual<DeviceOnlyStorage>();
  real_univ->bind(*real_univ_mem);
  auto real_univ_view = real_univ ->view();
  
  auto real_univ_buf = ctxt->make_contextual<StorageBuffer>(
    (StorageMeasure)[=]{ return real_univ->size(); },
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  auto real_univ_buf_mem = ctxt->make_contextual<StagingStorage>(
    StorageOptimization::Send);
  real_univ_buf->bind(*real_univ_buf_mem);


  auto sim_univs = ctxt->make_contextual<ColorAttachmentStorageImage>(
    VkExtent2D{ 360, 240 }, 1, VK_FORMAT_R32_SFLOAT);
  auto sim_univs_mem = ctxt->make_contextual<DeviceOnlyStorage>();
  sim_univs->bind(*sim_univs_mem);
  auto sim_univs_view = sim_univs->view();

  auto sim_univs_buf = ctxt->make_contextual<StorageBuffer>(
    (StorageMeasure)[=]{ return sim_univs->size(); },
    VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  auto sim_univs_buf_mem = ctxt->make_contextual<StagingStorage>(
    StorageOptimization::Fetch);
  sim_univs_buf->bind(*sim_univs_buf_mem);

  auto cost_buf = ctxt->make_contextual<StorageBuffer>(
    (StorageMeasure)[=]{ return sizeof(float); },
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  auto cost_buf_mem = ctxt->make_contextual<StagingStorage>(
    StorageOptimization::Fetch);
  cost_buf->bind(*cost_buf_mem);




  select_phys_dev(*ctxt);

  
  // Set up input data.
  DeformSpecs spec {};
  spec.rotate = 1.;
  spec.stretch = { 1., 1. };
  spec.translate = { 1., 1. };
  deform_in_mem->send(&spec, 0, sizeof(DeformSpecs));

  Bacterium bac {};
  bac.orient = 1.;
  bac.pos = { 0., 0. };
  bac.size = { 4., 5. };
  bac.univ = 0;
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


  std::vector<char> buf(sim_univs_buf->size(), 0.1);

  real_univ_buf_mem->send(buf.data(), 0, buf.size());

  eval->execute(deform_out_buf->view(),
    real_univ_buf->view(),
    *real_univ_view,
    sim_univs_buf->view(),
    *sim_univs_view,
    cost_buf->view());

  sim_univs_buf_mem->fetch(buf.data(), 0, buf.size());





  std::getc(stdin);
  return 0;
}
