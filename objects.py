import numpy as np
import simpy
import logging
import functools

GEMM_TIME = 100
SPM_READ_TIME = 1
SPM_WRITE_TIME = 1


def log_decorator(log_message):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            logging.log(16, f'[{self.env.now}]Simulator: Slice {
                        self.index} {log_message} start')
            yield self.env.process(func(self, *args, **kwargs))
            logging.log(16, f'[{self.env.now}]Simulator: Slice {
                        self.index} {log_message} end')
        return wrapper
    return decorator


class Slice:
    def __init__(self, env: simpy.Environment, index):
        self.env = env
        self.spm = ScratchPadMemory(env)
        self.index = index

    @log_decorator('spm_allocate')
    def spm_allocate(self, *args):
        self.spm.allocate(*args)
        yield self.env.timeout(0)

    @log_decorator('spm_free')
    def spm_free(self, id):
        self.spm.free(id)
        yield self.env.timeout(0)

    @log_decorator('load')
    def load(self, allocate_id, array):
        self.spm.read(allocate_id, array)
        yield self.env.timeout(SPM_WRITE_TIME)

    @log_decorator('store')
    def store(self, allocate_id, array):
        self.spm.write(allocate_id, array)
        yield self.env.timeout(SPM_WRITE_TIME)

    @log_decorator('gemm')
    def gemm(self, q, k, o):
        q_a, k_a, o_a = self.spm.get(q, k, o)
        np.matmul(q_a, k_a, out=o_a)
        yield self.env.timeout(GEMM_TIME)


class ScratchPadMemory():
    def __init__(self, env: simpy.Environment):
        self.env = env
        self.memory = {}

    def allocate(self, id, shape, dtype):
        array = np.zeros(shape, dtype=dtype)
        self.memory[id] = array

    def free(self, id):
        del self.memory[id]

    def read(self, id, data):
        data.array = self.memory[id]

    def write(self, id, data):
        self.memory[id] = data.array

    def get(self, *args):
        res = []
        for id in args:
            res.append(self.memory[id])
        return tuple(res)
