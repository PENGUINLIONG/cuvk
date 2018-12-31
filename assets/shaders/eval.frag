//
// Evaluation Shader Program (3/3)
// -------------------------------
//  In this shader stage, white color is given to all the pixels where there is
//  a bacterial cell.
//L
#version 450



//
// Output
// ------
//  The differences between simulated universes and the real universe.
layout(location=0)
out float sim_univ;
//L



void main() {
  sim_univ = 1.;
}
