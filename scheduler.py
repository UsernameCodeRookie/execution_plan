import simpy
import logging
from collections import deque
import numpy as np
from memory import TMA
from objects import Slice
from typing import List
from functools import partial


class Data():
    def __init__(self, array):
        self.array = array


class Barrier():
    def __init__(self, env: simpy.Environment):
        self.env = env
        self.barriers = [{} for _ in range(16)]
        self.stall = [False for _ in range(16)]
        self.barrier_queue = [deque() for _ in range(16)]

    def claim_bar(self, idx, bar_id):
        self.barriers[idx][bar_id] = 0
        yield self.env.timeout(0)

    def req_bar(self, idx, bar_list):
        for bar_id in bar_list:
            self.barriers[idx][bar_id] += 1
        yield self.env.timeout(0)

    def rel_bar(self, idx, bar_list):
        for bar_id in bar_list:
            self.barriers[idx][bar_id] -= 1
        yield self.env.timeout(0)

    def wait_bar(self, idx, bar_list):
        for bar_id in bar_list:
            if self.barriers[idx][bar_id] > 0:
                self.stall[idx] = True
                self.insert_wait_bar(idx, bar_list)
                break
        yield self.env.timeout(0)

    def insert_wait_bar(self, idx, bar_list):
        wait_bar = self.wait_bar(idx, bar_list)
        self.barrier_queue[idx].appendleft([wait_bar])

    def append(self, idx, program):
        self.barrier_queue[idx].append(program)

    def fetch(self, idx):
        return self.barrier_queue[idx].popleft()

    def empty(self, idx):
        return len(self.barrier_queue[idx]) == 0


class Scheduler():
    slices: List[Slice]
    tma: TMA
    barrier: Barrier

    def __init__(self, slices, tma, barrier):
        self.slices = slices
        self.tma = tma
        self.barrier = barrier

    def schedule(self, instr, args):
        runtime = []
        # not barrier instruction
        # directly execute
        if instr == 'make_tensor' or instr == 'spm_allocate' or instr == 'claim_barrier':
            task = self.not_barrier_instr(instr, args)
            runtime.append(task)
        # barrier instruction
        # push to barrier queue
        else:
            self.barrier_instr(instr, args)

        self.check_queue(runtime)

        return runtime

    def not_barrier_instr(self, instr, args):
        match instr:
            case 'claim_barrier':
                slice_idx, id, shape, dtype = args
                func = self.barrier.claim_bar(slice_idx, id)

            case 'make_tensor':
                dim_len, tensor_id, dim, tile_dim, dtype = args
                func = self.tma.make_tensor(tensor_id, dim, tile_dim, dtype)

            case 'spm_allocate':
                slice_idx, id, shape, dtype = args
                func = self.slices[slice_idx].spm_allocate(id, shape, dtype)

        task = [func]
        return task

    def check_queue(self, runtime):
        for i in range(16):

            if self.barrier.empty(i):
                continue

            task = self.barrier.fetch(i)
            # if wait bar is not satisfied, then insert back to queue
            runtime.append(task)

    def barrier_instr(self, instr, args):
        match instr:
            case 'spm_free':
                slice_idx, id = args
                free = self.slices[slice_idx].spm_free(id)
                self.barrier.append(slice_idx, [free])

            case 'tma_store_slice':
                slice_idx, id, tile, wait_bar_list = args
                tensor_id, tile_pos = tile

                # wait barrier
                wait_bar = self.barrier.wait_bar(slice_idx, wait_bar_list)

                array = Data(np.array([1]))
                slice_to_tma = self.slices[slice_idx].load(id, array)
                tma_to_ddr = self.tma.write_ddr(tensor_id, tile_pos, array)
                self.barrier.append(slice_idx, [wait_bar])
                self.barrier.append(slice_idx, [slice_to_tma, tma_to_ddr])

            case 'tma_load_multicast':
                tile, id, mask, set_bar_list = args
                tensor_id, tile_pos = tile

                req_bar_list = {}

                for i in mask:
                    req_bar = self.barrier.req_bar(i, set_bar_list)
                    req_bar_list[i] = req_bar

                array = Data(np.array([1]))
                ddr_to_tma = self.tma.read_ddr(tensor_id, tile_pos, array)

                for i in mask:
                    req_bar = req_bar_list[i]
                    tma_to_slice = self.slices[i].store(id, array)
                    rel_bar = self.barrier.rel_bar(i, set_bar_list)
                    self.barrier.append(
                        i, [req_bar, ddr_to_tma, tma_to_slice, rel_bar])

            case 'slice_gemm':
                slice_idx, template_args, q, k, v, wait_bar_list, set_bar_list = args

                wait_bar = self.barrier.wait_bar(slice_idx, wait_bar_list)
                req_bar = self.barrier.req_bar(slice_idx, set_bar_list)
                gemm = self.slices[slice_idx].gemm(q, k, v)
                rel_bar = self.barrier.rel_bar(slice_idx, set_bar_list)

                self.barrier.append(slice_idx, [wait_bar])
                self.barrier.append(slice_idx, [req_bar, gemm, rel_bar])


CPU_CYCLE_TIME = 1


class CPU:
    def __init__(self, env: simpy.Environment, program, scheduler: Scheduler):
        self.env = env
        self.program_iter = program
        self.scheduler = scheduler

    def loop(self):
        while True:
            yield self.env.timeout(CPU_CYCLE_TIME)
            program = next(self.program_iter)

            if program is None:
                break

            instr, args = program
            runtime = self.scheduler.schedule(instr, args)

            for task in runtime:
                self.env.process(self.run_task(task))

    def run_task(self, task):
        for func in task:
            yield self.env.process(func)
