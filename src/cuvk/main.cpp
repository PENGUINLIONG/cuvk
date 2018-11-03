#include <cuvk/context.hpp>
#include <cstdlib>

using namespace cuvk;

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
  Context ctxt;
  select_phys_dev(ctxt);

  return 0;
}
