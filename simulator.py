from objects import Slice
from memory import TMA
from program import Program
from scheduler import Scheduler, Barrier, CPU
import simpy
import logging


class Simulator():
    def __init__(self, file_path):
        self.env = simpy.Environment()
        self.slices = [Slice(self.env, i) for i in range(16)]
        self.tma = TMA(self.env)
        self.barrier = Barrier(self.env)

        self.program = Program(file_path, self.slices, self.tma)

        self.scheduler = Scheduler(self.slices, self.tma, self.barrier)
        self.cpu = CPU(self.env, self.program, self.scheduler)

    def run(self, simtime=None):
        self.env.run(until=simtime)

    def init(self):
        self.env.process(self.cpu.loop())


if __name__ == '__main__':
    logging.basicConfig(level=14, filename='resource/runtime.log',
                        filemode='w', format='%(message)s')
    sim = Simulator('program.txt')
    sim.init()
    sim.run()
