//
// Evaluation Shader Program (2/3)
// -------------------------------
//  In this shader stage, Cell meshes are generated using a simple model.
//       6 x-----------------------x 4
//       /                           \ 
//   8 x                               x 2
//     |                               |
// 10 x                                 x 1
//     |                               |
//   9 x                               x 3
//       \                           /
//       7 x-----------------------x 5
//L
#version 450
precision mediump float;



//
// Type Definitions
// ----------------
//  Bacterium descriptors.
struct Bacterium {
  // Center of the cell.
  vec2 pos;
  // Size (half length excluding the round tip, radius) of the bacterium.
  vec2 size;
  // Orientation of the cell, CCW from x-axis.
  float orient;
  // ID of universe the bacterium is in.
  uint univ;
};
//L



//
// Inputs
// -------
layout(points)
in;
//  Bacterium.
layout(location=0)
in Bacterium bacs[];
//L



//
// Output
// ------
//  Built-in position output variable.
layout(triangle_strip, max_vertices = 10)
out; // out vec2 gl_Position;
//  Built-in layer indicator output. Here we use it to pass universe IDs.
// out int gl_Layer;
//L



//
// Push Constants
// --------------
layout(std430, push_constant) uniform EvalMeta {
  // The index of the first universe.
  uint BASE_UNIV;
};
//L



vec4 calc_pos(mat2 rotate, vec2 orig, vec2 offset) {
  return vec4(rotate * orig + offset, 0.0, 1.0);
}

void main() {
  Bacterium bac = bacs[0];
  float len = bac.size.x;
  float r = bac.size.y;
  float trig_45_r = 0.70710678118654752440084436210485 * r;
  vec2 pos = bac.pos;

  float sin_o = sin(bac.orient);
  float cos_o = cos(bac.orient);
  mat2x2 rotate = { { cos_o, -sin_o }, { sin_o, cos_o } };
  
  // Right of the round tip.
  gl_Position = calc_pos(rotate, vec2(len + r, 0.0), pos);
  gl_PrimitiveID = gl_PrimitiveIDIn;
  EmitVertex();
  // Right top of the round tip.
  gl_Position = calc_pos(rotate, vec2(len + trig_45_r, trig_45_r), pos);
  gl_PrimitiveID = gl_PrimitiveIDIn;
  EmitVertex();
  // Right bottom of the round tip.
  gl_Position = calc_pos(rotate, vec2(len + trig_45_r, -trig_45_r), pos);
  gl_PrimitiveID = gl_PrimitiveIDIn;
  EmitVertex();
  // Right top of the cylindrical part.
  gl_Position = calc_pos(rotate, vec2(len, r), pos);
  gl_PrimitiveID = gl_PrimitiveIDIn;
  EmitVertex();
  // Right bottom of the cylindrical part.
  gl_Position = calc_pos(rotate, vec2(len, -r), pos);
  gl_PrimitiveID = gl_PrimitiveIDIn;
  EmitVertex();
  // Left top of the cylindrical part.
  gl_Position = calc_pos(rotate, vec2(-len, r), pos);
  gl_PrimitiveID = gl_PrimitiveIDIn;
  EmitVertex();
  // Left bottom of the cylindrical part.
  gl_Position = calc_pos(rotate, vec2(-len, -r), pos);
  gl_PrimitiveID = gl_PrimitiveIDIn;
  EmitVertex();
  // Left top of the round tip.
  gl_Position = calc_pos(rotate, vec2(-len - trig_45_r, trig_45_r), pos);
  gl_PrimitiveID = gl_PrimitiveIDIn;
  EmitVertex();
  // Left bottom of the round tip.
  gl_Position = calc_pos(rotate, vec2(-len - trig_45_r, -trig_45_r), pos);
  gl_PrimitiveID = gl_PrimitiveIDIn;
  EmitVertex();
  // Left of the round tip.
  gl_Position = calc_pos(rotate, vec2(-len - r, 0.0), pos);
  gl_PrimitiveID = gl_PrimitiveIDIn;
  EmitVertex();

  gl_Layer = int(bac.univ - BASE_UNIV);
  EndPrimitive();
}
