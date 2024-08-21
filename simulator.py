import numpy as np
import simpy


class Slice():
    def __init__(self, env):
        self.__env = env
        self.__spm = ScratchPadMemory(env)
        self.__barriers = {}

    class Barrier():
        def __init__(self, env, rows, cols):
            self.rows = rows
            self.cols = cols
            self.__res = simpy.Resource(env)

        def has_request(self):
            return self.__res.count != 0

        def request(self):
            return self.__res.request()

    def spm_allocate(self, *args):
        self.__spm.allocate(*args)

    def claim_barrier(self, barrier_id, rows, cols):
        self.__barriers[barrier_id] = self.Barrier(self.__env, rows, cols)

    def load_set_barrier(self, allocate_id, array, barrier_id):
        with self.__barriers[barrier_id].request() as req:
            yield req

            yield self.__env.process(self.__spm.write(allocate_id, array))

    def store_wait_barrier(self, allocate_id, barrier_id):
        while self.__barriers[barrier_id].has_request():
            yield self.__env.timeout(1)
        yield self.__env.process(self.__spm.read(allocate_id))

    def gemm(self, q, k, o, wait_barrier_id, set_barrier_id):
        while self.__barriers[wait_barrier_id].has_request():
            yield self.__env.timeout(1)

        with self.__barriers[set_barrier_id].request() as req:
            yield req
            q_a, k_a, o_a = self.__spm.get(q, k, o)
            np.matmul(q_a, k_a, out=o_a)


class ScratchPadMemory():
    def __init__(self, env):
        self.__env = env
        self.__memory = {}

    def allocate(self, id, rows, cols, dtype):
        array = np.zeros((rows, cols), dtype=dtype)
        self.__memory[id] = array

    def read(self, id):
        print(f'{self.__env.now}: spm read {id} start')
        yield self.__env.timeout(10)
        print(f'{self.__env.now}: spm read {id} end')
        return self.__memory[id]

    def write(self, id, array):
        print(f'{self.__env.now}: spm write {id} start')
        yield self.__env.timeout(10)
        print(f'{self.__env.now}: spm write {id} end')
        self.__memory[id] = array

    def get(self, *args):
        res = []
        for id in args:
            res.append(self.__memory[id])
        return tuple(res)
