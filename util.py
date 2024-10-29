import numpy as np
from parser import parser
import itertools
from tqdm import tqdm
import random


def parse_tensor(state):
    instr, args = parser.parse(state)[0]

    for i, arg in enumerate(args):
        if isinstance(arg, tuple):
            _, args[i] = arg

    return args


def sample_tensor(state):
    args = parse_tensor(state)

    dim_len, tensor_id, dim, tile_dim, dtype = args

    fp = np.memmap(f'resource/{tensor_id}.npy',
                   dtype=dtype, mode='r', shape=tuple(dim))

    iter_space = list(map(lambda e, f: e // f, dim, tile_dim))

    iteration = itertools.product(*[range(i) for i in iter_space])

    sample_pos = random.sample(list(iteration), 1)[0]

    print('Sampled tile:', sample_pos)

    slices = get_slices(sample_pos, tile_dim)

    print(fp[slices])


def generate_tensor(state):
    args = parse_tensor(state)

    dim_len, tensor_id, dim, tile_dim, dtype = args

    generate_random_data(f'resource/{tensor_id}.npy', dim, tile_dim, dtype)


def get_slices(tile_pos, tile_shape):
    start_pos = list(map(lambda e, f: e * f, tile_pos, tile_shape))
    end_pos = list(map(lambda e, f: e + f, start_pos, tile_shape))

    slices = tuple(slice(s, e) for s, e in zip(start_pos, end_pos))
    return slices


def random_tile(fp, pos, dimention, tile_dimention, dtype):

    for i, d in enumerate(pos):
        if d >= dimention[i]:
            # out of bound
            return

    slices = get_slices(pos, tile_dimention)
    rand = np.random.rand(*tile_dimention).astype(dtype)
    fp[slices] = rand


def generate_random_data(file_path, dimention, tile_dimention, dtype):
    iter_space = list(map(lambda e, f: e // f, dimention, tile_dimention))
    fp = np.memmap(file_path, dtype=dtype, mode='w+', shape=tuple(dimention))
    # generate random data

    total = np.prod(iter_space)

    print(f'Generating random data for {file_path} with shape {
          dimention} and tile shape {tile_dimention}, iteration space {iter_space}')

    iteration = itertools.product(*[range(i) for i in iter_space])

    for indices in tqdm(iteration, total=total, desc=file_path):
        random_tile(fp, indices, dimention, tile_dimention, dtype)

    fp.flush()


def generate_random_q_and_k(q, k):
    generate_tensor(q)
    generate_tensor(k)


if __name__ == '__main__':
    with open('resource/program.txt', 'r') as file:
        lines = file.readlines()
        q = lines[1]
        k = lines[2]
        o = lines[3]

        generate_random_q_and_k(q, k)
        # sample_tensor(o)
