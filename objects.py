import numpy as np
import simpy

GEMM_TIME = 10


class Slice():
    def __init__(self, env: simpy.Environment):
        self.env = env
        self.spm = ScratchPadMemory(env)
        self.barriers = {}

    class Barrier():
        def __init__(self, env, rows, cols):
            self.rows = rows
            self.cols = cols
            self.res = simpy.Resource(env)

        def has_request(self):
            return self.res.count != 0

        def request(self):
            return self.res.request()

    def spm_allocate(self, *args):
        self.spm.allocate(*args)

    def claim_barrier(self, barrier_id, rows, cols):
        self.barriers[barrier_id] = self.Barrier(self.env, rows, cols)

    def load_set_barrier(self, allocate_id, array, barrier_id):
        with self.barriers[barrier_id].request() as req:
            yield req

            yield self.env.process(self.spm.write(allocate_id, array))

    def store_wait_barrier(self, allocate_id, barrier_id):
        while self.barriers[barrier_id].has_request():
            yield self.env.timeout(1)
        yield self.env.process(self.spm.read(allocate_id))

    def gemm(self, q, k, o, wait_barrier_id, set_barrier_id):
        while self.barriers[wait_barrier_id].has_request():
            yield self.env.timeout(1)

        with self.barriers[set_barrier_id].request() as req:
            yield req
            q_a, k_a, o_a = self.spm.get(q, k, o)
            np.matmul(q_a, k_a, out=o_a)
            yield self.env.timeout(GEMM_TIME)


SPM_READ_TIME = 10
SPM_WRITE_TIME = 10


class ScratchPadMemory():
    def __init__(self, env: simpy.Environment):
        self.env = env
        self.memory = {}

    def allocate(self, id, rows, cols, dtype):
        array = np.zeros((rows, cols), dtype=dtype)
        self.memory[id] = array

    def read(self, id):
        print(f'{self.env.now}: spm read {id} start')
        yield self.env.timeout(SPM_READ_TIME)
        print(f'{self.env.now}: spm read {id} end')
        return self.memory[id]

    def write(self, id, array):
        print(f'{self.env.now}: spm write {id} start')
        yield self.env.timeout(SPM_WRITE_TIME)
        print(f'{self.env.now}: spm write {id} end')
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

    def run_program(self, program_iterator):
        while True:
            yield self.env.timeout(CPU_CYCLE_TIME)
            program = next(program_iterator)
            for instr in program:
                yield self.env.process(instr)


class TMA():
    def __init__(self, env: simpy.Environment):
        self.env = env
