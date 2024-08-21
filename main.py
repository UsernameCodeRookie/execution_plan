import simpy
import numpy as np
from simulator import Slice, ScratchPadMemory


env = simpy.Environment()
slice = Slice(env)
slice.spm_allocate('spm', 4, 4, 'float16')
slice.claim_barrier('bar', 4, 4)
slice.claim_barrier('bar_not', 4, 4)

env.process(slice.load_set_barrier(
    'spm', np.ones((4, 4), dtype='float16'), 'bar'))
env.process(slice.store_wait_barrier('spm', 'bar_not'))

env.run(until=100)
