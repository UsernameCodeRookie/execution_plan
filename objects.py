import numpy as np
import simpy
import logging

GEMM_TIME = 100
SPM_READ_TIME = 10
SPM_WRITE_TIME = 10


class Slice():
    def __init__(self, env: simpy.Environment, index):
        self.env = env
        self.spm = ScratchPadMemory(env)
        self.barriers = {}
        self.index = index

    class Barrier():
        def __init__(self, env, shape, dtype):
            self.shape = shape
            self.dtype = dtype
            self.res = simpy.Resource(env)

        def has_request(self):
            return self.res.count != 0

        def request(self):
            self.req = self.res.request()
            return self.req

        def release(self):
            return self.res.release(self.req)

    def spm_allocate(self, *args):
        logging.log(16, f'[{self.env.now}]Simulator: Slice {
            self.index} spm_allocate {args}')
        self.spm.allocate(*args)
        yield self.env.timeout(0)

    def claim_barrier(self, barrier_id, shape, dtype):
        logging.log(16, f'[{self.env.now}]Simulator: Slice {
            self.index} claim_barrier {barrier_id}')
        self.barriers[barrier_id] = self.Barrier(self.env, shape, dtype)
        yield self.env.timeout(0)

    def barrier_request(self, barrier_id_list):
        for barrier_id in barrier_id_list:
            yield self.barriers[barrier_id].request()

    def barrier_release(self, barrier_id_list):
        for barrier_id in barrier_id_list:
            yield self.barriers[barrier_id].release()

    def barrier_wait(self, barrier_id_list):
        while any([self.barriers[barrier_id].has_request() for barrier_id in barrier_id_list]):
            yield self.env.timeout(1)

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

    def read(self, id, array):
        array = self.memory[id]

    def write(self, id, array):
        self.memory[id] = array

    def get(self, *args):
        res = []
        for id in args:
            res.append(self.memory[id])
        return tuple(res)


CPU_CYCLE_TIME = 1


class CPU():
    def __init__(self, env: simpy.Environment):
        self.env = env

    def run(self, program_iterator):
        while True:
            yield self.env.timeout(CPU_CYCLE_TIME)
            self.env.process(self.run_program(program_iterator))

    def run_program(self, program_iterator):

        program = next(program_iterator)

        if program is None:
            return

        for instr in program:
            yield self.env.process(instr)
