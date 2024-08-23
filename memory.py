import simpy
import numpy as np


class TMA():
    def __init__(self, env: simpy.Environment):
        self.env = env

    def make_tensor(self):
        pass

    def read_ddr(self):
        pass

    def write_ddr(self):
        pass


DDR_READ_TIME = 0
DDR_WRITE_TIME = 0


class DDR():
    def __init__(self, env: simpy.Environment):
        self.env = env

    def make_tensor(self, dimension, tile_dimension, dtype):
        shape = list(map(lambda e, f: e * f, dimension, tile_dimension))
        self.tile_shape = tile_dimension
        self.mmap = np.memmap('ddr.npy', dtype=dtype,
                              mode='w+', shape=shape)

        yield self.env.timeout(0)

    def read(self, tile_pos, array):
        pos = list(map(lambda e, f: e * f, tile_pos, self.tile_shape))
        array[:] = self.mmap[pos[0]:pos[0] + self.tile_shape[0],
                             pos[1]:pos[1] + self.tile_shape[1]]

        yield self.env.timeout(DDR_READ_TIME)


if __name__ == '__main__':
    print(list(map(lambda e, f: e * f, [4, 3, 2048, 128], [1, 1, 64, 128])))
