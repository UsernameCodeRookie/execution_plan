from objects import Slice
from memory import TMA
from execution_plan import ExecutionPlan
from scheduler import Scheduler, Barrier, ExecutionPlanManager
import simpy
import logging


class Simulator():
    def __init__(self, file_path):
        self.env = simpy.Environment()
        self.slices = [Slice(self.env, i) for i in range(16)]
        self.tma = TMA(self.env)
        self.barrier = Barrier(self.env)

        self.plan = ExecutionPlan(file_path)

        self.scheduler = Scheduler(self.slices, self.tma, self.barrier)
        self.elm = ExecutionPlanManager(self.env, self.plan, self.scheduler)

    def run(self, simtime=None):
        self.env.run(until=simtime)

    def init(self):
        self.env.process(self.elm.loop())


if __name__ == '__main__':
    logging.basicConfig(level=14, filename='resource/runtime.log',
                        filemode='w', format='%(message)s')
    sim = Simulator('program.txt')
    sim.init()
    sim.run()
