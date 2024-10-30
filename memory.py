import simpy
import numpy as np
import logging
import os
import functools

DDR_READ_TIME = 10
DDR_WRITE_TIME = 10


def log_decorator(log_message):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            logging.log(16, f'[{self.env.now}]Simulator: {
                        log_message.format(*args)} start')
            result = func(self, *args, **kwargs)
            if isinstance(result, simpy.events.Event):
                yield self.env.process(result)
            else:
                return result
            logging.log(16, f'[{self.env.now}]Simulator: {
                        log_message.format(*args)} end')
        return wrapper
    return decorator


class TMA():
    def __init__(self, env: simpy.Environment):
        self.env = env
        self.ddr = DDR(env)

    @log_decorator('TMA make Tensor {0}, dimension {1}, tile_dimension {2}, dtype {3}')
    def make_tensor(self, *args):
        self.ddr.make_tensor(*args)
        yield self.env.timeout(0)

    @log_decorator('TMA read DDR, Tensor {0}, Tile {1}')
    def read_ddr(self, *args):
        yield self.env.timeout(DDR_READ_TIME)
        self.ddr.read(*args)

    @log_decorator('TMA write DDR, Tensor {0}, Tile {1}')
    def write_ddr(self, *args):
        yield self.env.timeout(DDR_WRITE_TIME)
        self.ddr.write(*args)


class DDR():
    def __init__(self, env: simpy.Environment):
        self.env = env
        self.tensor_map = {}
        self.tile_shape = {}

    def make_tensor(self, tensor_id, dimension, tile_dimension, dtype):
        self.tile_shape[tensor_id] = tile_dimension

        path = os.path.join('resource', f'{tensor_id}.npy')

        mmap = np.memmap(path, dtype=dtype,
                         mode='w+', shape=tuple(dimension))
        self.tensor_map[tensor_id] = mmap

    def read(self, tensor_id, tile_pos, data):
        mmap = self.tensor_map[tensor_id]
        data.array = mmap[self.get_slices(tile_pos, tensor_id)]

    def write(self, tensor_id, tile_pos, data):
        mmap = self.tensor_map[tensor_id]
        mmap[self.get_slices(tile_pos, tensor_id)] = data.array[:]

    def position_complete(self, tile_pos):
        return [1 for _ in range(len(self.tile_shape) - len(tile_pos))] + tile_pos

    def get_slices(self, tile_pos, tensor_id):
        tile_pos = self.position_complete(tile_pos)

        start_pos = list(map(lambda e, f: e * f, tile_pos,
                         self.tile_shape[tensor_id]))
        end_pos = list(map(lambda e, f: e + f, start_pos,
                       self.tile_shape[tensor_id]))

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
