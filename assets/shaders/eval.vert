//
// Evaluation Shader Program (1/3)
// -------------------------------
//  In this shader stage, cells are placed into different universes.
//L
#version 450
precision mediump float;



//
// Type Definitions
// ----------------
//  Bacteria descriptors.
struct Bac {
  // Center of the cell.
  vec2 pos;
  // Size (half length excluding the round tip, radius) of the bacterium.
  vec2 size;
  // Orientation of the cell, CCW from x-axis.
  float orient;
  // ID of universe the bacterium is in.
  int univ;
};
//L



//
// Inputs
// ------
//  Bacterium.
// Center of the cell.
layout(location=0) 
in vec2 pos;
// Size (half length excluding the round tip, radius) of the bacterium.
layout(location=1)
in vec2 size;
// Orientation of the cell, CCW from x-axis.
layout(location=2)
in float orient;
//  ID of universe the bacterium is in.
layout(location=3)
in int univ;
//L



//
// Outputs
// -------
//  Bacteria.
layout(location=4)
out Bac bac;
//L



void main() {
  bac = Bac(pos, size, orient, univ);
}
