#include <stdint.h>

#ifndef L_CUVK_H
#define L_CUVK_H

// # CUVK C-API
//
// This is the exposed CUVK C-APIs for language bindings.
//
// ## 1 Convention
//
// All exposed APIs follow the following conventions.
//
// ### 1.1 Call Convention
//
// All exported CUVK functions use the standard call convention as normal stack
// behavior.
//
#ifndef _MSC_VER
#define __declspec(x)
#endif

#ifdef L_CUVK_COMPILE
#define L_EXPORT extern "C" __declspec(dllexport)
#else
#define L_EXPORT extern "C" __declspec(dllimport)
#endif

#ifdef _MSC_VER
#define L_STDCALL __stdcall
#else
#define L_STDCALL __attribute__((stdcall))
#endif
//
// ### 1.2 Function parameters
//
// Function parameters without explicit attribution are input-only.
//
// Parameters attributed with `L_OUT` are output parameters.
//
#define L_OUT
//
// Parameters attributed with `L_INOUT` will be used as input at first, a new
// value will be assigned at the end of the call.
//
#define L_INOUT
//
// ## 2 Primitive Type Definitions
//
// Two primitive types are used for interfacing. Note that here we only discuss
// about types of C-APIs' parameters and return values. For arrange of data
// transferred to GPU, please refer to `shader_interface.hpp`.
//
typedef uint32_t CuvkBool;
typedef uint32_t CuvkSize;
//
// ## 3 Error Handling
//
// CUVK functions report error through return values. A failed call will return
// `false`. Otherwise, `true` is returned.
//
// A CUVK call fails when the funcation is unable to finish execution for non-
// recoverable reasons. The corresponding CUVK component is invalidated. An
// invalidated component will not be able to function as specified. All invalid
// components *must not* be used.
typedef CuvkBool CuvkResult;
//
// # 4 Logging
//
// By default, log is written to standard error. To specify output file path:
//
L_EXPORT CuvkResult L_STDCALL cuvkRedirectLog(
  const char* path
);
//
// ## 5 Synchronization
//
// All CUVK components are synchronized internally.
//
// ## 6 Initialization
//
// CUVK *must* be initialized before any invocation to other functions. A Vulkan
// instance is created so that CUVK can make consequential calls to Vulkan APIs.
//
// ### 6.1 Initialize CUVK
//
L_EXPORT CuvkResult L_STDCALL cuvkInitialize(
  CuvkBool debug
);
//
// Fails when:
// - The required Vulkan version is not supported by the Vulkan runtime.
//
// **WARN** Extra validations will be done when debug mode is enabled. This
// *may* slow down most of the Vulkan calls.
//
// ### 6.2 Deinitialize CUVK
//
L_EXPORT void L_STDCALL cuvkDeinitialize();
//
// ## 7 Contexts
//
// Context is an abstraction of Vulkan devices, where CUVK tasks are executed
// and memories are allocated. CUVK *can* have multiple contexts.
typedef struct CuvkContextInfo {} *CuvkContext;
//
// ### 7.1 Enumerate Physical Devices
//
// Physical device is a representaion of an existing Vulkan device. Every CUVK
// context is built on a physical device. To enumerate physical devices that
// satisfy CUVK's requirements:
//
L_EXPORT void L_STDCALL cuvkEnumeratePhysicalDevices(
  L_INOUT CuvkSize* jsonSize,
  L_OUT char* pJson
);
//
// If the parameter `jsonSize` is 0, then the size of string buffer required to
// contain `json` is set to it. If `jsonSize` is greater than 0, the json of
// length `jsonSize` is written to `json`.
//
// Sample code for enumeration:
//
// ```cpp
// CuvkSize size;
// std::string phys_dev_info;
// cuvkEnumeratePhysicalDevices(nullptr, &size);
// phys_dev_info.resize(static_cast<size_t>(size));
// cuvkEnumeratePhysicalDevices(&size, &phys_dev_info);
// ```
//
// ### 7.2 Context Creation
//
// Memory allocation is done internally by CUVK during context creation. The
// user-application *must* provide this requirement so that CUVK can calculate
// how much memory should be allocated.
struct CuvkMemoryRequirements {
  // Number of deformation specifications.
  CuvkSize nspec;
  // Number of bacteria input per batch. The bacteria output in deformation
  // tasks and input in evaluation tasks will have size `nspecs * nbac`.
  CuvkSize nbac;
  // Number of universes to be renderd in one batch. This number will also be
  // the length of cost output.
  CuvkSize nuniv;
  // Width of the simulated and the real universes.
  CuvkSize width;
  // Height of the simulated and the real universes.
  CuvkSize height;
};
L_EXPORT CuvkResult L_STDCALL cuvkCreateContext(
  CuvkSize physicalDeviceIndex,
  L_INOUT CuvkMemoryRequirements* memoryRequirements,
  L_OUT CuvkContext* pContext
);
//
// Fails when:
// - The device is unable to fulfill the memory requirements.
//
//
// ### 7.3 Context Destruction
//
// Context must be destroyed if it is nolonger used. The user application *must*
// ensure all components rely on the context is release before calling to this
// function.
//
L_EXPORT void L_STDCALL cuvkDestroyContext(
  CuvkContext context
);
//
// ## 8 Tasks
//
// Tasks in CUVK are seperated into two parts: task creation and task execution.
//
// Task creation involves the filling of command buffer, which is a relatively
// slow process. CUVK create tasks asynchronously, a transient handle is
// returned.
//
// Task execution involves the communication with Vulkan devices, including data
// transfer and shader program execution.
//
// The user application must wait CUVK to finish the task. A handle to the task
// is returned as result.
//
typedef struct CuvkTaskInfo {} *CuvkTask;
//
// **NOTE** Host memories *must* be kept alive until the invocation is finished.
//
// ### 8.1 Task Creation
//
// Tasks are followed by immediate execution of that task. CUVK currently
// provide the following task types:
//
// - Deformation
// - Evaluation
//
// #### 8.1.1 Deformation
//
// In the deformation stage, bacteria sets are expanded according to a set of
// deformation specification.
//
struct CuvkDeformationInvocation {
  // Deformation specification data buffer. Deform specs are not updated if this
  // field is `nullptr`.
  const void* pDeformSpecs;
  // Number of deformation specification in `pDeformSpecs`.
  CuvkSize nSpec;
  // Bacteria data buffer.
  const void* pBacs;
  // Number of bacteria in `pBacteria`.
  CuvkSize nBac;
  // The minimum universe ID what will be added to cells' original universeID.
  CuvkSize baseUniv;
  // The maximum universe ID occurred in the bacteria data + 1.
  CuvkSize nUniv;
  // Deformed bacteria as output.
  L_OUT void* pBacsOut;
};
L_EXPORT CuvkResult L_STDCALL cuvkInvokeDeformation(
  CuvkContext context,
  const CuvkDeformationInvocation* pInvocation,
  L_OUT CuvkTask* pTask
);
//
// Fails when:
// - Unexpected failure occurs.
//
// #### 8.1.2 Evaluation
//
// In evaluation stage, bacteria are drawn to universes.
//
struct CuvkEvaluationInvocation {
  // Bacteria data buffer. If this field is `nullptr` input data is directly
  // taken from the output of deformation stage.
  const void* pBacs;
  // Number of bacteria in `pBacs`.
  CuvkSize nBac;
  // Width of the simulated and the real universes. Must use the same value as
  // that used to create CUVK context, otherwise it will lead to undefined
  // behavior.
  CuvkSize width;
  // Height of the simulated and the real universes. Must use the same value as
  // that used to create CUVK context, otherwise it will lead to undefined
  // behavior.
  CuvkSize height;
  // Simulated universes.
  L_OUT void* pSimUnivs;
  // Real universe.
  const void* pRealUniv;
  // Number of universes in `pSimUnivs`.
  CuvkSize nSimUniv;
  // ID of the fisrt universe in simulated universes.
  CuvkSize baseUniv;
  // Costs.
  L_OUT void* pCosts;
};
L_EXPORT CuvkResult L_STDCALL cuvkInvokeEvaluation(
  CuvkContext context,
  const CuvkEvaluationInvocation* pInvocation,
  L_OUT CuvkTask* pTask
);
//
// Fails when:
// - Unexpected failure occurs.
//
// **NOTE** The invocation will not check if all the bacteria are in the drawn
// universes.
//
// ## 8.3 Polling
//
// CUVK allow user applications to manage task execution stati flexibly by
// employing the polling model.
//
enum CuvkTaskStatus {
  CUVK_TASK_STATUS_NOT_READY = 0,
  CUVK_TASK_STATUS_OK        = 1,
  CUVK_TASK_STATUS_ERROR     = 2,
};
L_EXPORT CuvkTaskStatus L_STDCALL cuvkPoll(
  CuvkTask task
);
//
// **WARN** If a task failed to complete, the result is invalid. User
// applications *must not* rely on results of failed execution.
//
// ## 8.4 Task Destruction
//
// Every task must be destructed when unused. The user application *should*
// ensure the task has completed; otherwise this call will block the current
// thread until the task becomes completed.
//
L_EXPORT void L_STDCALL cuvkDestroyTask(
  CuvkTask task
);

#endif // !L_CUVK_H
