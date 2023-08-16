# Copyright (c) 2023, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view


cdef extern from "nvtext/bpe_tokenize.hpp" namespace "nvtext" nogil:

    cdef struct bpe_merge_pairs "nvtext::bpe_merge_pairs":
        pass

    cdef unique_ptr[bpe_merge_pairs] load_merge_pairs_file(
        const string &filename_merges
    ) except +

    cdef unique_ptr[column] byte_pair_encoding(
        const column_view &strings,
        const bpe_merge_pairs &merge_pairs
    ) except +
