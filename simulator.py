from objects import Slice, CPU
from program import CpuIterator
import simpy
import logging


class Simulator():
    def __init__(self):
        self.env = simpy.Environment()
        self.slices = [Slice(self.env, i) for i in range(16)]
        self.cpu = CPU(self.env)

        self.program = CpuIterator(self.slices)

    def run(self, simtime=100):
        self.env.run(until=simtime)

    def process(self, event):
        self.env.process(event)

    def init(self):
        self.env.process(self.cpu.run_program(self.program))


if __name__ == '__main__':
    logging.basicConfig(level=0, format='%(message)s')
    sim = Simulator()
    sim.init()
    sim.run()
