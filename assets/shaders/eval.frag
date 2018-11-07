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
// Uniform Variables
// -----------------
//  The real next universe.
layout(r32f, binding=0) readonly
uniform image2D real_univ;
//L



//
// Output
// ------
//  The differences between simulated universes and the real universe.
layout(location=0)
out float sim_univ;
//L

void main() {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  float real_val = imageLoad(real_univ, coord).x;
  sim_univ = 1.;
}
