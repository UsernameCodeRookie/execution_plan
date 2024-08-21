from objects import Slice, CPU
from program import ProgramIterator
import simpy


class Simulator():
    def __init__(self):
        self.env = simpy.Environment()
        self.slices = [Slice(self.env) for _ in range(16)]
        self.cpu = CPU(self.env)

        self.program = ProgramIterator(self.slices)

    def run(self, simtime=100):
        self.env.run(until=simtime)

    def process(self, event):
        self.env.process(event)

    def init(self):
        self.env.process(self.cpu.run_program(self.program))
