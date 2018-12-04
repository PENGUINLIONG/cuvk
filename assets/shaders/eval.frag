//
// Evaluation Shader Program (3/3)
// -------------------------------
//  In this shader stage, white color is given to all the pixels where there is
//  a bacterial cell.
//L
#version 450



//
// Input
// -----
//  Built-in input of layer indicater. Here we use it to fetch universe IDs.
//in int gl_Layer;
//  Built-in variable of the current pixel coordination.
layout(origin_upper_left)
in vec4 gl_FragCoord;
//L



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
