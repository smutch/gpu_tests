cdef extern from "init_cuda.h":
    void c_init_cuda "init_cuda"()

def init_cuda():
    c_init_cuda()
