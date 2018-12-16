# CUVK

CellUniverse Vulkan implementation. This distribution is used to accelerate cell deformation, universe rendering and cost calculation.

## Work in Progress

**CUVK is still in progress.** The current implementation is constrained by the capability of compute devices. Future works will aim at providing work-arounds under the limits of the device when viable.

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
