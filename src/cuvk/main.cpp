#include <cuvk/shader_interface.hpp>
#include <cuvk/cuvk.h>
#include <cuvk/logger.hpp>
#include <fmt/format.h>
#include <cstdlib>

using namespace cuvk::shader_interface;

int main() {
  cuvkInitialize(true);
  CuvkMemoryRequirements mem_req {};
  mem_req.width = 16;
  mem_req.height = 16;
  mem_req.nbac = 1;
  mem_req.nspec = 1;
  mem_req.nuniv = 1;
  mem_req.shareBacteriaBuffer = true;
  CuvkContext ctxt;
  cuvkCreateContext(0, mem_req, &ctxt);

  DeformSpecs spec {};
  spec.rotate = 3.1415926 / 2.;
  spec.stretch = { 1., 1. };
  spec.translate = { -0.2, -0.1 };

  Bacterium bac {};
  bac.orient = 0.5;
  bac.pos = { 0.5, 0.5 };
  bac.size = { 0.5, 0.5 };
  bac.univ = 1;

  Bacterium bac_out {};

  DeformationInvocation deform_invoke {};
  deform_invoke.nSpec = 1;
  deform_invoke.nBac = 1;
  deform_invoke.pDeformSpecs = &spec;
  deform_invoke.pBacs = &bac;
  deform_invoke.pBacsOut = &bac_out;

  std::array<float, 16 * 16> real_univ;
  for (auto& v : real_univ) {
    v = 1.;
  }
  /*
  real_univ[0] = 2. / 3.;
  real_univ[255] = 2. / 3.;
  */
  std::array<float, 16 * 16> sim_univ {};

  float cost = 0.;

  EvaluationInvocation eval_invoke {};
  eval_invoke.baseUniv = 1;
  eval_invoke.nBac = 1;
  eval_invoke.nSimUniv = 1;
  eval_invoke.pBacs = &bac;
  eval_invoke.pCosts = &cost;
  eval_invoke.pRealUniv = real_univ.data();
  eval_invoke.pSimUnivs = sim_univ.data();
  eval_invoke.realUnivSize = 16 * 16 * sizeof(float);

  CuvkTask deform_task, eval_task;
  cuvkInvokeDeformation(ctxt, &deform_invoke, &deform_task);
  while (cuvkPoll(deform_task) == CUVK_TASK_STATUS_NOT_READY) {}
  cuvkInvokeEvaluation(ctxt, &eval_invoke, &eval_task);
  while (cuvkPoll(eval_task) == CUVK_TASK_STATUS_NOT_READY) {}



  cuvk::LOG.trace("deformed cell: "
    "pos=({}, {}), size=({},{}), orient={}, univ={}",
    bac_out.pos[0], bac_out.pos[1],
    bac_out.size[0], bac_out.size[1],
    bac_out.orient,
    bac_out.univ);

  std::getc(stdin);

  return 0;
}
