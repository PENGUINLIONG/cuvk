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

  auto deform = ctxt->make_contextual<Deformation>();

  auto specs_aligned = 
    ctxt->get_aligned_size(sizeof(DeformSpecs), BufferType::StorageBuffer);
  auto bacs_aligned =
    ctxt->get_aligned_size(sizeof(Bacterium), BufferType::StorageBuffer);
  auto bacs_out_aligned =
    ctxt->get_aligned_size(sizeof(Bacterium), BufferType::StorageBuffer);

  auto deform_in_buf = ctxt->make_contextual<StorageBuffer>(
    specs_aligned + bacs_aligned, ExecType::Compute,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  auto deform_out_buf = ctxt->make_contextual<StorageBuffer>(
    bacs_out_aligned, ExecType::Compute,
    VK_BUFFER_USAGE_TRANSFER_DST_BIT);

  auto deform_in_mem = ctxt->make_contextual<Storage>(
    deform_in_buf->alloc_size(),
    StorageOptimization::Send);
  auto deform_out_mem = ctxt->make_contextual<Storage>(
    deform_out_buf->alloc_size(),
    StorageOptimization::Fetch);

  deform_in_buf->bind(deform_in_mem, 0);
  deform_out_buf->bind(deform_out_mem, 0);

  // Set up input data.
  DeformSpecs spec;
  spec.rotate = 0.;
  spec.stretch = { 1., 1. };
  spec.translate = { 0., 0. };
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
  deform_out_mem->invalidate_mapped(0, VK_WHOLE_SIZE);
  deform_out_mem->fetch(&bac_out, 0, sizeof(Bacterium));

  LOG.info("Deformed bactrium: orient={}, pos=({}, {}), size=({}, {})",
    bac_out.orient,
    bac_out.pos[0], bac_out.pos[1],
    bac_out.size[0], bac_out.size[1]);

  std::getc(stdin);
  return 0;
}
