from ctypes import *

class MemoryRequirements(Structure):
    _fields_ = [('nspec', c_uint),
                ('nbac', c_uint),
                ('nuniv', c_uint),
                ('width', c_uint),
                ('height', c_uint),
                ('share_bacteria_buffer', c_uint)]

class DeformationInvocation(Structure):
    _fields_ = [('deform_specs', c_buffer),
                ('nspec', c_uint),
                ('bacs', c_buffer),
                ('nbac', c_uint),
                ('bacs_out', c_buffer)]

class EvaluationInvocation(Structure):
    _fields_ = [('bacs', c_buffer),
                ('nbac', c_uint),
                ('sim_univs', c_buffer),
                ('real_univ', c_buffer),
                ('real_univ_size', c_uint),
                ('nsim_univ', c_uint),
                ('base_univ', c_uint),
                ('costs', c_buffer)]

LIBCUVK = cdll.LoadLibrary("libcuvk.dll")

def enumerate_physical_devices():
    LIBCUVK.cuvkEnumeratePhysicalDevices()

class Cuvk:
    def __init__()