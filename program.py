from parser import parser

# batch
BATCH_SIZE = 50


class Data():
    def __init__(self, array):
        self.array = array


class Program():

    def __init__(self, path):
        self.file = open('resource/' + path, 'r')
        self.res_batch = []

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

        for i, arg in enumerate(args):
            if isinstance(arg, tuple):
                _, args[i] = arg

        # return self.parse_program(instr, args)
        return instr, args

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
