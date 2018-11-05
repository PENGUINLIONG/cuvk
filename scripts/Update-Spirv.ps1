Remove-Item -Path './assets/shaders/*.spv'
Get-ChildItem -Path './assets/shaders' | ForEach-Object {
  glslangValidator $($_.FullName) -V -o $($_.FullName + '.spv')
}
