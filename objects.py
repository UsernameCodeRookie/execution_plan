import numpy as np
import simpy
import logging

GEMM_TIME = 100
SPM_READ_TIME = 1
SPM_WRITE_TIME = 1


class Slice():
    def __init__(self, env: simpy.Environment, index):
        self.env = env
        self.spm = ScratchPadMemory(env)
        self.index = index

    def spm_allocate(self, *args):
        logging.log(16, f'[{self.env.now}]Simulator: Slice {
            self.index} spm_allocate {args}')
        self.spm.allocate(*args)
        yield self.env.timeout(0)

    def spm_free(self, id):
        logging.log(16, f'[{self.env.now}]Simulator: Slice {
            self.index} spm_free {id}')
        self.spm.free(id)
        yield self.env.timeout(0)

    def load(self, allocate_id, array):
        logging.log(16, f'[{self.env.now}]Simulator: Slice {self.index} load {
            allocate_id} start')
        self.spm.read(allocate_id, array)
        yield self.env.timeout(SPM_WRITE_TIME)
        logging.log(16, f'[{self.env.now}]Simulator: Slice {self.index} load {
            allocate_id} end')

    def store(self, allocate_id, array):
        logging.log(16, f'[{self.env.now}]Simulator: Slice {self.index} store {
            allocate_id} start')
        self.spm.write(allocate_id, array)
        yield self.env.timeout(SPM_WRITE_TIME)
        logging.log(16, f'[{self.env.now}]Simulator: Slice {self.index} store {
            allocate_id} end')

    def gemm(self, q, k, o):
        logging.log(16, f'[{self.env.now}]Simulator: Slice {
            self.index} gemm {q} {k} {o} start')
        q_a, k_a, o_a = self.spm.get(q, k, o)
        np.matmul(q_a, k_a, out=o_a)
        yield self.env.timeout(GEMM_TIME)
        logging.log(16, f'[{self.env.now}]Simulator: Slice {
                    self.index} gemm {q} {k} {o} end')


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
