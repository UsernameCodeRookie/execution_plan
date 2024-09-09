from objects import Slice
from parser import parser
from functools import partial
from typing import List
from memory import TMA
import logging
import numpy as np

# batch
BATCH_SIZE = 50


class CpuIterator():

    def __init__(self, *args):

        self.file = open('resource/' + args[0], 'r')
        self.slices = args[1]
        self.tma = args[2]
        self.res_batch = []

        self.slices: List[Slice]
        self.tma: TMA

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.res_batch) == 0:
            instr_batch = self.read_file(BATCH_SIZE)

            if instr_batch is None:
                return None

            parse_result = parser.parse(instr_batch)

            if parse_result is None:
                return None
                # raise SyntaxError('Invalid syntax in program')

            self.res_batch = parse_result

        instr, args = self.res_batch.pop(0)

        return self.parse_program(instr, args)

    def read_file(self, size):
        lines = []
        for _ in range(size):
            line = self.file.readline()
            if not line:
                if lines:
                    return ''.join(lines)
                else:
                    return None
            elif line.startswith('-'):
                continue
            lines.append(line)
        return ''.join(lines)

    def parse_program(self, instr, args):
        program = []

        # parse args
        for i, arg in enumerate(args):
            if isinstance(arg, tuple):
                _, args[i] = arg

        match instr:
            case 'make_tensor':
                dim_len, tensor_id, dim, tile_dim, dtype = args
                logging.log(8, f'Program: Make Tensor {tensor_id} with dimension {
                            dim} and tile dimension {tile_dim} of type {dtype}')
                func = self.tma.make_tensor(tensor_id, dim, tile_dim, dtype)
                program.append(func)

            case 'spm_allocate':
                slice_idx, id, shape, dtype = args
                logging.log(8, f'Program: SPM {slice_idx} allocate {
                    id} with shape {shape} of type {dtype}')
                self.slices[slice_idx].spm_allocate(id, shape, dtype)
                func = self.slices[slice_idx].spm_allocate(
                    id, shape, dtype)
                program.append(func)

            case 'spm_free':
                slice_idx, id = args
                logging.log(8, f'Program: SPM {slice_idx} free {id}')
                func = self.slices[slice_idx].spm_free(id)
                program.append(func)

            case 'claim_barrier':
                slice_idx, id, shape, dtype = args
                logging.log(8, f'Program: Slice {slice_idx} claim Barrier {
                    id} with shape {shape} of type {dtype}')
                func = self.slices[slice_idx].claim_barrier(
                    id, shape, dtype)
                program.append(func)

            case 'tma_store_slice':
                slice_idx, id, tile, wait_bar = args
                tensor_id, tile_pos = tile

                logging.log(8, f'Program: TMA store from Slice {slice_idx} to Tensor {tensor_id} Tile {
                    tile_pos} and wait Barrier {wait_bar}')

                array = np.array([1])

                wait_barrier = self.slices[slice_idx].barrier_wait(wait_bar)
                program.append(wait_barrier)

                slice_to_tma = self.slices[slice_idx].store(id, array)
                tma_to_ddr = self.tma.write_ddr(tensor_id, tile_pos, array)
                program.append(slice_to_tma)
                program.append(tma_to_ddr)

            case 'tma_load_multicast':
                tile, id, mask, set_bar = args
                tensor_id, tile_pos = tile

                logging.log(8, f'Program: TMA load multicast from Tensor {tensor_id} Tile {
                    tile_pos} to Slice {mask} and set Barrier {set_bar}')

                array = np.array([1])

                for i in mask:
                    barrier_request = self.slices[i].barrier_request(set_bar)
                    program.append(barrier_request)

                slice_to_tma = self.tma.read_ddr(tensor_id, tile_pos, array)
                program.append(slice_to_tma)
                for i in mask:
                    tma_to_slice = self.slices[i].load(id, array)
                    program.append(tma_to_slice)
                    barrier_release = self.slices[i].barrier_release(set_bar)
                    program.append(barrier_release)

            case 'slice_gemm':
                slice_idx, template_args, q, k, v, wait_bar_list, set_bar_list = args

                logging.log(8, f'Program: Slice {slice_idx} GEMM with template args {template_args}, q {
                            q} k{k} v{v}, wait Barrier {wait_bar_list} set Barrier {set_bar_list}')

                barrier_wait = self.slices[slice_idx].barrier_wait(
                    wait_bar_list)
                barrier_request = self.slices[slice_idx].barrier_request(
                    set_bar_list)
                gemm = self.slices[slice_idx].gemm(
                    q, k, v)
                barrier_release = self.slices[slice_idx].barrier_release(
                    set_bar_list)
                program.append(barrier_request)
                program.append(barrier_wait)
                program.append(gemm)
                program.append(barrier_release)

        return program


if __name__ == '__main__':
    slices = [Slice(None, i) for i in range(16)]
    program = CpuIterator(slices)
    for func_batch in program:
        for func, args in func_batch:
            func()
