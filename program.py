from objects import *
from parser import parser

# batch
BATCH_SIZE = 50


class Program():

    def __init__(self, *args):

        self.file = open('program.txt', 'r')
        self.slices = args[0]
        self.res_batch = []

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.res_batch) == 0:
            instr_batch = self.read_file(BATCH_SIZE)
            parse_result = parser.parse(instr_batch)

            if parse_result is None:
                raise SyntaxError('Invalid syntax in program')

            self.res_batch = parse_result

        instr, args = self.res_batch.pop(0)
        match instr:
            case 'spm_allocate':
                slice_idx, id, rows, cols, dtype = args
                print(f'Program: SPM {slice_idx} allocate {
                      id} with shape {rows}x{cols} of type {dtype}')
                self.slices[slice_idx].spm_allocate(id, rows, cols, dtype)

    def read_file(self, size):
        lines = []
        for _ in range(size):
            line = self.file.readline()
            if not line:
                if lines:
                    return ''.join(lines)
                else:
                    self.file.close()
                    raise StopIteration
            lines.append(line)
        return ''.join(lines)


if __name__ == '__main__':
    slices = [Slice(None) for _ in range(16)]
    program = Program(slices)
    next(program)
