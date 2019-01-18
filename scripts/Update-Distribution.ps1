# Remove previous version of distribution.
if (Test-Path ./dist) {
    Remove-Item ./dist -Recurse -Force
}

# Compile shaders.
scripts/Update-Spirv.ps1

# Compile CUVK.
if (-not(Test-Path ./build)) {
    New-Item build -ItemType Directory -Force
}
Set-Location build
cmake ..
cmake --build . --target libcuvk -- "-property:Configuration=Release"
Set-Location ..

# Copy files.
New-Item -Path "dist/include","dist/lib","dist/assets/shaders","dist/scripts" -ItemType Directory -Force
Copy-Item -Path "python/*" -Destination "dist/python" -Recurse
Copy-Item -Path "include/cuvk/cuvk.h" -Destination "dist/include"
Copy-Item -Path "build/Release/libcuvk.dll" -Destination "dist/lib"
Copy-Item -Path "build/Release/libcuvk.lib" -Destination "dist/lib"
Copy-Item -Path "README.md" -Destination "dist"
Copy-Item -Path "assets/shaders/*.spv" -Destination "dist/assets/shaders"
Copy-Item -Path "scripts/Run-Benchmark.ps1" -Destination "dist/scripts/Run-Benchmark.ps1"
