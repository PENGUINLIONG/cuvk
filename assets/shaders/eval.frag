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
layout(r32f, binding=1) writeonly
uniform image2DArray sim_univs;
//  Calculated costs.
layout(binding=2)
buffer costs_buf {
  float[] costs;
};
//L



void main() {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  float real_val = imageLoad(real_univ, coord).x;
  imageStore(sim_univs, ivec3(coord, gl_Layer), vec4(1.0, 0.0, 0.0, 0.0));
  costs[gl_Layer] += abs(real_val - 1.0);
}
