from objects import Slice
from parser import parser
from functools import partial
from typing import List
from memory import TMA
import logging
import numpy as np

# batch
BATCH_SIZE = 50


class Data():
    def __init__(self, array):
        self.array = array


class Program():

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
