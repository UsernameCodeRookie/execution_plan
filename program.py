from objects import *
from parser import parser

# batch
BATCH_SIZE = 50


class Program():

    def __init__(self, *args):

        self.file = open('program.txt', 'r')
        # self.slices = args[0]
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
                pass

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
    program = Program()
    next(program)
