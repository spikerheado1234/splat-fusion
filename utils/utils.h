#pragma once

#include <random>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cudnn.h>
#include "../ACSR/layout/acsr.h"


// Custom data-structures for launching kernels.
struct params {
    // Classic parameters.
    int b,s,n,h;
    // Parameters of the structurally sparse matrices.
    int w,d;

    // Put any other auxiliary variables into here as you see fit.
};

// Feed this as a function input.
struct inps {

    // Device side allocated arrays.
    float * inp_one;
    float * inp_two;
    float * answer;

    int one_size; int two_size;

    int answer_size;
};

// Container for host arrays.
struct host_inps {
  float *inp_one;
  float *inp_two;
  float *answer;
  int one_size;
  int two_size;
  int ans_size;
};

#define CHECK_CUDNN_ERR(func) 	\
{				\
  cudnnStatus_t status = (func);                   \
  if (status != CUDNN_STATUS_SUCCESS) {                                      \
    printf("CUDNN API failed at line %d file: %s, with error: %s (%d)\n",      \
           __LINE__,__FILE__, cudnnGetErrorString(status), status);           \
    exit(EXIT_FAILURE);                                             \
  }                                                                 \
}

#define CHECK_CUDA(func)                                            \
{                                                                   \
  cudaError_t status = (func);                                      \
  if (status != cudaSuccess) {                                      \
    printf("CUDA API failed at line %d file: %s, with error: %s (%d)\n",      \
           __LINE__,__FILE__, cudaGetErrorString(status), status);           \
    exit(EXIT_FAILURE);                                             \
  }                                                                 \
}

#define CHECK_CUSPARSE(func)                                        \
{                                                                   \
  cusparseStatus_t status = (func);                                 \
  if (status != CUSPARSE_STATUS_SUCCESS) {                          \
    printf("CUSPARSE API failed at line %d with error: %s (%d)\n",  \
           __LINE__, cusparseGetErrorString(status), status);       \
    exit(EXIT_FAILURE);                                             \
  }                                                                 \
}

#define IF_SUCCESS(stat, msg) \
	if (stat != cudaSuccess) { \
		printf(msg); \
		std::cout << "\nError is: " << cudaGetErrorString(stat) << std::endl; \
		return EXIT_FAILURE; \
	}

#define IF_SUCCESS_BLAS(stat, msg) \
	if (stat != CUBLAS_STATUS_SUCCESS) { \
		printf(msg); \
		std::cout << cublasGetStatusString(stat) << std::endl; \
		return EXIT_FAILURE; \
	}

inline std::chrono::time_point<std::chrono::high_resolution_clock> time_now() {
  return std::chrono::high_resolution_clock::now();
}

inline int64_t time_elapsed_ms(std::chrono::time_point<std::chrono::high_resolution_clock> start) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(time_now() - start).count();
}

inline int64_t time_elapsed_us(std::chrono::time_point<std::chrono::high_resolution_clock> start) {
  return std::chrono::duration_cast<std::chrono::microseconds>(time_now() - start).count();
}

void print_transformation(transformations* transforms, int sequence_length, int block_height);
void matrix_fill(float* matrix, int size);

// Simple printing of a 2-d tensor.
void print_matrix(float**matrix, int outer, int dim1, int dim2, int ld);

// Simple printing of a 4-d tensor.
void print_matrix(float* matrix, int batch_size, int sequence_length, int num_heads, int hidden_dimension);
void print_matrix(int* matrix, int batch_size, int sequence_length, int num_heads, int hidden_dimension);

// Transpose tensor dimensions dim1 & dim2.
void transpose(const float*tensor, size_t d0, size_t d1, size_t d2, size_t d3, float * output, size_t dim1, size_t dim2);

/// Fill array with random number between uniform distribution [0, 1].
void fill_rand(float *A, size_t size);

///// Fill a square mask matrix with the column sparsity pattern with gap `d`.
void fill_mask_column(float *mask, int s, int d);

///// Fill the mask with a blocked sparsity pattern.
void fill_mask_blocked(float * mask, int s, int w);

///// Fill the mask with a windowed sparsity pattern.
void fill_mask_windowed(float * mask, int s, int w);
//
///// Fill the mask with a strided sparsity pattern.
void fill_mask_strided(float * mask, int s, int w);

///// Test that all values in the two arrays are the same (within tolerance).
bool allclose(const float *A, const float *B, size_t size, float tolerance = 1e-6);

///// Generalized transpose of two axes of a 4-D tensor of shape [d0, d1, d2, d3] and
///// stored in row-major order. Regular transpose is equivalent to transposing
///// the last two dimensions (dim1=2, dim2=3).
void transpose(const float *tensor, size_t d0, size_t d1, size_t d2, size_t d3,
               float *output, size_t dim1, size_t dim2);

/// Sampled Dense-Dense Matrix Multiply (SDDMM) for 4D tensors, where inputs are
///// treated as batched 2D matrices and mask is also 2D.
///// - A shape    [d0 x d1 x d2 x d3]
///// - B shape    [d0 x d1 x d3 x d2]
///// - c shape    [d0 x d1 x d2 x d2]
///// - mask shape [d2 x d2]
void batch_sddmm(const float *A, const float *B, const float *mask,
                                size_t d0, size_t d1, size_t d2, size_t d3, float *C);

/// Convert dense 2D matrix to sparse CSR format. Returns number of nonzeros.
/// `ind`, `col`, and `val` are out-parameters and caller is responsible for
/// memory allocated for these.
size_t dense2csr(const float *A, size_t d0, size_t d1, int **ind, int **col, float **val);

size_t dense2csr_batched(const float *A, size_t d0, size_t d1, size_t batch_dim, int **ind, int **col, float **val);

// Convert a dense one-d tensor: [bnsh] into a two-d tensor [bn, sh]
// for use in APIs like cublas SGEMM.
void un_flatten(float * mat_one, float *** mat_two, size_t batch, size_t seq, size_t heads, size_t hid_dim);