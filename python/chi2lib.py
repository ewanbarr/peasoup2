import ctypes as C
lib = C.CDLL("./pychi2lib.so")
candidate_sigma = lib.py_candidate_sigma
candidate_sigma.argtypes = [C.c_double,C.c_int,C.c_double,C.c_bool]
candidate_sigma.restype = C.c_double
power_for_sigma = lib.py_power_for_sigma
power_for_sigma.argtypes = [C.c_double,C.c_int,C.c_double,C.c_bool]
power_for_sigma.restype= C.c_double

