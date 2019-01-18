# CUVK

CellUniverse Vulkan implementation. This distribution is used to accelerate cell deformation, universe rendering and cost calculation.

## Work in Progress

**CUVK is still in progress.** The current implementation is constrained by the capability of compute devices. Future works will aim at providing work-arounds under the limits of the device when viable.

## Performance

The Python demo program (Release build) produced the following result on _my machine_, compared with the adapted CPU-based implementation:

### Deformation

|Run#      |Baseline (ms)|CUVK (ms)|%Reduction|
|----------|-------------|---------|----------|
|0         |       13.752|    2.426|      82.4|
|1         |       13.159|    2.534|      80.7|
|2         |       16.747|    5.942|      64.5|
|3         |       16.451|    5.076|      69.1|
|4         |       13.140|    2.781|      78.8|
|5         |       13.886|    5.267|      62.1|
|6         |       13.067|    5.512|      57.8|
|7         |       13.266|    5.469|      58.8|
|8         |       13.364|    5.343|      60.0|
|9         |       14.516|    2.463|      83.0|
|**Median**|   **13.558**|**5.172**|  **61.9**|

### Evaluation

|Run#      |Baseline (ms)|CUVK (ms)   |%Reduction|
|----------|-------------|------------|----------|
|0         |    10117.638|    2837.001|      72.0|
|1         |     8291.345|    2267.195|      72.6|
|2         |     8286.471|    2315.688|      72.1|
|3         |     8356.441|    2252.020|      73.1|
|4         |     8386.348|    2284.402|      72.8|
|5         |     8289.388|    2230.294|      73.1|
|6         |     8403.162|    2238.656|      73.4|
|7         |     8395.716|    2287.414|      72.8|
|8         |     8387.057|    2246.775|      73.2|
|9         |     8289.592|    2192.331|      73.6|
|**Median**| **8371.395**|**2259.608**|  **73.0**|

### How to Reproduce

A handy PowerShell script `scripts/Run-Benchmark.ps1` can be used to reproduce the result. Notice that the CPU-based algorithm is adapted to mock the API of CUVK. This can make the baseline different from the original implementation.

NOTE: _my machine_ is defined as following:

* CPU: Intel Core i7 8650U
* RAM: 16GB 1866MHz PDDR3
* GPU: NVIDIA GeForce GTX 1060
* GRAM: 6GB GDDR5

## C-API

CUVK's raw C-API and detailed documentation is covered in the header file `include/cuvk/cuvk.h`. Language bindings (e.g. for Java) can be created based on the C-API.

The following functions are exported:

* `cuvkRedirectLog` (*NOT IMPLEMENTED YET*) Redirect the logging stream to a file. By default the log in printed via standard error.
* `cuvkInitialize` Initialize CUVK.
* `cuvkDeinitialize` Deinitialize CUVK and release all resources.
* `cuvkEnumeratePhysicalDevices` (*NOT IMPLEMENTED YET*) Enumerate all physical device information in JSON. It can be helpful to choose which physical device to use when there are multiple Vulkan-enabled devices.
* `cuvkCreateContext` Create a context on the physical device and allocate all resources needed for computation and get a handle of it.
* `cuvkDestroyContext` Destroy the context with all related resources released.
* `cuvkInvokeDeformation` Creat, dispatch a deformation task and get a handle to the result.
* `cuvkInvokeEvaluation` Create, dispatch an evaluation task and get a handle to the result.
* `cuvkPoll` Poll a task, i.e., check if the task is finished, and if it's successfully finished.
* `cuvkDestroyTask` Destroy the task and release related resources.

**NOTE** Task creation in CUVK is asynchronous. Returning from invocation of a task neither imply that the Vulkan device has received the instructions, nor the data provided is transfered to the device. All host-owned resources must be kept alive until the poll returned a *finished* state (either `OK` or `ERROR`). Release data before task completion can lead to undefined behavior.

## Python Language Binding

An naive Python language binding is attached in `python/cuvk.py`. Please refer to the demo program `python/demo.py` to see how to use it.
