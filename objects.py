import numpy as np
import simpy
import logging

GEMM_TIME = 10


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

    # def load_set_barrier(self, allocate_id, barrier_id, array):
    #     logging.log(0, f'[{self.env.now}]Simulator: Slice {self.index} load_set_barrier {
    #           allocate_id} {barrier_id}')
    #     with self.barriers[barrier_id].request() as req:
    #         yield req

    #         yield self.env.process(self.spm.write(allocate_id, array))

    def load_set_barrier(self, allocate_id, barrier_id_list, array):
        logging.log(16, f'[{self.env.now}]Simulator: Slice {self.index} load_set_barrier {
            allocate_id} {barrier_id_list}')
        # with self.barriers[barrier_id].request() as req:
        #     yield req

        #     yield self.env.process(self.spm.write(allocate_id, array))
        for barrier_id in barrier_id_list:
            yield self.barriers[barrier_id].request()

        yield self.env.process(self.spm.write(allocate_id, array))

        for barrier_id in barrier_id_list:
            yield self.barriers[barrier_id].release()

    # def store_wait_barrier(self, allocate_id, barrier_id, array):
    #     logging.log(0, f'[{self.env.now}]Simulator: Slice {self.index} store_wait_barrier {
    #           allocate_id} {barrier_id}')
    #     while self.barriers[barrier_id].has_request():
    #         yield self.env.timeout(1)
    #     yield self.env.process(self.spm.read(allocate_id, array))

    def store_wait_barrier(self, allocate_id, barrier_id_list, array):
        logging.log(16, f'[{self.env.now}]Simulator: Slice {self.index} store_wait_barrier {
            allocate_id} {barrier_id_list}')
        while any([self.barriers[barrier_id].has_request() for barrier_id in barrier_id_list]):
            yield self.env.timeout(1)
        yield self.env.process(self.spm.read(allocate_id, array))

    def gemm(self, q, k, o, wait_barrier_id_list, set_barrier_id_list):
        while any([self.barriers[wait_barrier_id].has_request() for wait_barrier_id in wait_barrier_id_list]):
            yield self.env.timeout(1)

        for set_barrier_id in set_barrier_id_list:
            yield self.barriers[set_barrier_id].request()

        q_a, k_a, o_a = self.spm.get(q, k, o)
        np.matmul(q_a, k_a, out=o_a)
        yield self.env.timeout(GEMM_TIME)

        for set_barrier_id in set_barrier_id_list:
            yield self.barriers[set_barrier_id].release()


SPM_READ_TIME = 10
SPM_WRITE_TIME = 10


class ScratchPadMemory():
    def __init__(self, env: simpy.Environment):
        self.env = env
        self.memory = {}

    def allocate(self, id, shape, dtype):
        array = np.zeros(shape, dtype=dtype)
        self.memory[id] = array

    def read(self, id, array):
        logging.log(16, f'[{self.env.now}]Simulator: spm read {id} start')
        yield self.env.timeout(SPM_READ_TIME)
        logging.log(16, f'[{self.env.now}]Simulator: spm read {id} end')
        array = self.memory[id]

    def write(self, id, array):
        logging.log(16, f'[{self.env.now}]Simulator: spm write {id} start')
        yield self.env.timeout(SPM_WRITE_TIME)
        logging.log(16, f'[{self.env.now}]Simulator: spm write {id} end')
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
