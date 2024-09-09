from objects import Slice, CPU
from memory import TMA
from program import CpuIterator
import simpy
import logging


class Simulator():
    def __init__(self, file_path):
        self.env = simpy.Environment()
        self.slices = [Slice(self.env, i) for i in range(16)]
        self.cpu = CPU(self.env)
        self.tma = TMA(self.env)

        self.program = CpuIterator(file_path, self.slices, self.tma)

    def run(self, simtime=None):
        self.env.run(until=simtime)

    def process(self, event):
        self.env.process(event)

    def init(self):
        self.env.process(self.cpu.run(self.program))


if __name__ == '__main__':
    logging.basicConfig(level=14, filename='resource/runtime.log',
                        filemode='w', format='%(message)s')
    sim = Simulator('program.txt')
    sim.init()
    sim.run()
