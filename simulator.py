import numpy as np
import threading


class Slice():
    def __init__(self):
        self.__spm = ScratchPadMemory()
        self.__barriers = {}

    class Barrier():
        def __init__(self, rows, cols):
            self.rows = rows
            self.cols = cols
            self.__locked = False
            self.__lock = threading.Lock()

        def lock(self):
            self.__locked = True
            self.__lock.acquire()

        def unlock(self):
            self.__locked = False
            self.__lock.release()

        def is_locked(self):
            return self.__locked

        def sync(self):
            self.__lock.acquire()
            self.__lock.release()

    def claim_barrier(self, barrier_id, rows, cols):
        self.__barriers[barrier_id] = self.Barrier(rows, cols)

    def load_set_barrier(self, allocate_id, array, barrier_id):
        self.__barriers[barrier_id].lock()

        self.__spm.write(allocate_id, array)

        self.__barriers[barrier_id].unlock()

    def store_wait_barrier(self, allocate_id, barrier_id):
        # while not self.__barriers[barrier_id].is_locked():
        #     pass
        self.__barriers[barrier_id].sync()

        return self.__spm.read(allocate_id)

    def gemm(self, q, k, o, wait_barrier_id, set_barrier_id):
        # while not self.__barriers[wait_barrier_id].is_locked():
        #     pass
        self.__barriers[wait_barrier_id].sync()

        self.__barriers[set_barrier_id].lock()

        q_a, k_a, o_a = self.__spm.get(q, k, o)
        np.matmul(q_a, k_a, out=o_a)

        self.__barriers[set_barrier_id].unlock()


class ScratchPadMemory():
    def __init__(self):
        self.__memory = {}

    def allocate(self, id, rows, cols, dtype):
        array = np.zeros((rows, cols), dtype=dtype)
        self.__memory[id] = array

    def read(self, id):
        return self.__memory[id]

    def write(self, id, array):
        self.__memory[id] = array

    def get(self, *args):
        res = []
        for id in args:
            res.append(self.__memory[id])
        return tuple(res)
