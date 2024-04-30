#pragma once

template<class T> __global__ void blocked_kernel(T* queries, T* keys, T* values, T* answer, T* l, T* m, int sparsity_param, int batch, int num_heads, int seq_length, int hidden_dim);
