import simpy
import numpy as np
import logging
import tempfile
import os
import shutil


DDR_READ_TIME = 20
DDR_WRITE_TIME = 20


class TMA():
    def __init__(self, env: simpy.Environment):
        self.env = env
        self.ddr = DDR(env)

    def make_tensor(self, *args):
        logging.log(16, f'[{self.env.now}]Simulator: TMA make Tensor {
                    args[0]}, dimension {args[1]}, tile_dimension {args[2]}, dtype {args[3]}')
        self.ddr.make_tensor(*args)
        yield self.env.timeout(0)

    def read_ddr(self, *args):
        logging.log(16, f'[{self.env.now}]Simulator: TMA read DDR, Tensor {
                    args[0]}, Tile {args[1]} start')
        yield self.env.timeout(DDR_READ_TIME)
        self.ddr.read(*args)
        logging.log(16, f'[{self.env.now}]Simulator: TMA read DDR, Tensor {
                    args[0]}, Tile {args[1]} end')

    def write_ddr(self, *args):
        logging.log(16, f'[{self.env.now}]Simulator: TMA write DDR, Tensor {
                    args[0]}, Tile {args[1]} start')
        yield self.env.timeout(DDR_WRITE_TIME)
        self.ddr.write(*args)
        logging.log(16, f'[{self.env.now}]Simulator: TMA write DDR, Tensor {
                    args[0]}, Tile {args[1]} end')


class DDR():
    def __init__(self, env: simpy.Environment):
        self.env = env
        self.tensor_map = {}

    def make_tensor(self, tensor_id, dimension, tile_dimension, dtype):
        shape = list(map(lambda e, f: e * f, dimension, tile_dimension))
        self.tile_shape = tile_dimension

        path = os.path.join('resource', f'{tensor_id}.npy')

        mmap = np.memmap(path, dtype=dtype,
                         mode='w+', shape=shape)
        self.tensor_map[tensor_id] = mmap

    def read(self, tensor_id, tile_pos, array):
        mmap = self.tensor_map[tensor_id]
        array = mmap[self.get_slices(tile_pos)]

    def write(self, tensor_id, tile_pos, array):
        mmap = self.tensor_map[tensor_id]
        mmap[self.get_slices(tile_pos)] = array[:]

    def position_complete(self, tile_pos):
        return [1 for _ in range(len(self.tile_shape) - len(tile_pos))] + tile_pos

    def get_slices(self, tile_pos):
        tile_pos = self.position_complete(tile_pos)

        start_pos = list(map(lambda e, f: e * f, tile_pos, self.tile_shape))
        end_pos = list(map(lambda e, f: e + f, start_pos, self.tile_shape))

        slices = tuple(slice(s, e) for s, e in zip(start_pos, end_pos))
        return slices


if __name__ == '__main__':
    print(list(map(lambda e, f: e * f, [4, 3, 2048, 128], [1, 1, 64, 128])))
    ddr = DDR(None)
    ddr.make_tensor('q', [4, 3, 2048, 128], [1, 1, 64, 128], np.float32)
    array = np.ones([64, 128], dtype=np.float32)
    ddr.write('q', [0, 0, 0, 0], array)

    array_2 = np.zeros([64, 128], dtype=np.float32)
    print(array_2)
    ddr.read('q', [0, 0, 0, 0], array_2)
    print(array_2)
