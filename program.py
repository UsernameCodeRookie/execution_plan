from objects import *
from parser import parser
from functools import partial
from typing import List

# batch
BATCH_SIZE = 50


class ProgramIterator():

    def __init__(self, *args):

        self.file = open('program.txt', 'r')
        self.slices = args[0]
        self.res_batch = []

        self.slices: List[Slice]

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
                    self.file.close()
                    return None
            lines.append(line)
        return ''.join(lines)

    def parse_program(self, instr, args):
        program = []

        match instr:
            case 'spm_allocate':
                slice_idx, id, rows, cols, dtype = args
                print(f'Program: SPM {slice_idx} allocate {
                      id} with shape {rows}x{cols} of type {dtype}')
                self.slices[slice_idx].spm_allocate(id, rows, cols, dtype)
                func = self.slices[slice_idx].spm_allocate(
                    id, rows, cols, dtype)
                program.append(func)

            case 'claim_barrier':
                slice_idx, id, rows, cols, dtype = args
                print(f'Program: Slice {slice_idx} claim Barrier {
                      id} with shape {rows}x{cols} of type {dtype}')
                func = self.slices[slice_idx].claim_barrier(
                    id, rows, cols, dtype)
                program.append(func)

            case 'tma_store_slice':
                slice_idx, id, tile, wait_bar = args
                _, wait_bar = wait_bar  # ('wait_bar', id)
                tensor_id, tile_pos = tile

                print(f'Program: TMA store from Slice {slice_idx} to Tensor {tensor_id} Tile {
                      tile_pos} and wait Barrier {wait_bar}')

                array = None
                slice_to_tma = self.slices[slice_idx].store_wait_barrier(
                    id, wait_bar, array)
                program.append(slice_to_tma)

            case 'tma_load_multicast':
                tile, id, mask, set_bar = args
                tensor_id, tile_pos = tile
                _, mask = mask
                _, set_bar = set_bar

                print(f'Program: TMA load multicast from Tensor {tensor_id} Tile {
                      tile_pos} to Slice {mask} and set Barrier {set_bar}')

                array = None
                for i in mask:
                    tma_to_slice = self.slices[i].load_set_barrier(
                        id, set_bar, array)
                    program.append(tma_to_slice)

        return program


if __name__ == '__main__':
    slices = [Slice(None, i) for i in range(16)]
    program = ProgramIterator(slices)
    for func_batch in program:
        for func, args in func_batch:
            func()
