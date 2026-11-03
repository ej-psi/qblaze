import math
import cmath
import numpy
import qblaze

import pytest


TESTS = [
    (1, [(0, 1j)]),
    (4, [(1, 1j)]),
    (6000, [(1, 1j / 2**0.5), (5000, 1 / 2**0.5)]),
    (2**20, [((i << 8) + (i * 37 % 256), cmath.exp(2j * math.pi * i / 2**12) / 2**6) for i in range(2**12)]),
    (2**20, [(2**20-4, 0.5), (2**20-3, -0.5j), (2**20-2, -0.5), (2**20-1, 0.5j)]),
]

@pytest.mark.parametrize("n, values", TESTS)
def test_import(n, values):
    state = numpy.zeros(n, dtype=numpy.complex128)
    for (i, v) in values:
        state[i] = v
    sim = qblaze.Simulator()
    sim.import_amplitudes(state)
    assert list(sim) == values


TESTS_SPARSE = [
    {0: 1j},
    {1 << 63: 1j},
    {1: 1j / 2**0.5, 5000: 1 / 2**0.5},
    {2**20-4: 0.5, 2**20-3: -0.5j, 2**20-2: -0.5, 2**20-1: 0.5j},
    {(123 << 100) | 1: -1.0},
]

def _sparse_state_dict_to_numpy(sparse_state):
    # Convert a dict a sparse state vector, even for more than 64 quits
    mask_64 = (1 << 64) - 1
    max_qubits = max(1, int(max(sparse_state.keys())).bit_length())
    index_buf = []
    keys = numpy.array([int(k) for k in sparse_state.keys()], dtype=object)
    mask64 = ~numpy.uint64(0)
    for i in range((max_qubits // 64) + 1):
        key_bits1 = (keys >> (i * 64))
        key_bits1 &= mask64
        key_bits2 = key_bits1.astype(numpy.uint64)
        index_buf.append(key_bits2)
    index_buf = numpy.array(index_buf)
    amp_buf = numpy.array(list(sparse_state.values()), dtype=numpy.complex128)
    return [index_buf,amp_buf]

@pytest.mark.parametrize("values", TESTS_SPARSE)
def test_import_sparse(values):
    sim = qblaze.Simulator()
    values_np_format = _sparse_state_dict_to_numpy(values)
    sim.import_amplitudes_sparse(values_np_format)
    sim.flush()
    assert dict(sim) == values
