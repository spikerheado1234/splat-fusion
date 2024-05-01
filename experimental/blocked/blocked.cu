#include "blocked.cuh"
#include <cmath>
#include "../../utils/utils.h"
#include "cuda_runtime.h"
#include <cstring>
#include <cassert>
#include <iostream>

#define BLOCK_SIZE_X 16 // This is b_c. -> this is also our coarsening factor.
#define BLOCK_SIZE_Y 16 // This is b_r.
#define INNER_DIM 8 // This is the inner dim we use in all the mat-muls. 
// Ensure that INNER_DIM is always smaller than Min(BLOCK_SIZE_X, BLOCK_SIZE_Y).

#define CHECK_CUDA(func)                                            \
{                                                                   \
  cudaError_t status = (func);                                      \
  if (status != cudaSuccess) {                                      \
    printf("CUDA API failed at line %d file: %s, with error: %s (%d)\n",      \
           __LINE__,__FILE__, cudaGetErrorString(status), status);           \
    exit(EXIT_FAILURE);                                             \
  }                                                                 \
}

__device__ bool blocked_is_computed(int row, int col, int sparsity_parameter) {
    // Figure out what block we are in.
    int block_num = row / sparsity_parameter;

    if (block_num < 1) {
        return col < sparsity_parameter;
    }

    // Otherwise we have some check to make.
    if (((block_num - 1) * sparsity_parameter) < col && col < (block_num * sparsity_parameter)) {
        return true;
    }

    return false;
}

// Queries -> row-major, Keys^T -> row-major, Values -> row-major. [batch, num_heads, seq_length, hidden_dim].
template<class T>
__global__ void blocked_kernel(T* queries, T* keys, T* values, T* answer, T * l, T * m, int sparsity_param, int batch, 
                            int num_heads, int seq_length, int hidden_dim) {

    int head_hidden_dim = hidden_dim / num_heads;

    // For much better memory coalescing, we need to index as follows:
    #define idx_queries(b,s,n,h) (((b)*num_heads+(n))*seq_length+(s))*head_hidden_dim+(h)
    #define idx_keys(b,s,n,h) (((b)*num_heads+(n))*head_hidden_dim+(h))*seq_length+(s)
    #define idx_values(b,s,n,h) (((b)*num_heads+(n))*seq_length+(s))*head_hidden_dim+(h)
    #define idx_output(b,s,n,h) (((b)*num_heads+(n))*seq_length+(s))*head_hidden_dim+(h)

    int batch_num = blockIdx.z / batch;
    int head_num = blockIdx.z % batch;

    int tx = threadIdx.x; int ty = threadIdx.y;
    int bx = blockIdx.x; int by = blockIdx.y;
    int row = by * blockDim.y + ty; int col = bx * blockDim.x + tx;

    // THis is for the first inner loop, computing the QiKj product.
    __shared__ T queries_shmem[BLOCK_SIZE_Y][INNER_DIM];
    __shared__ T keys_shmem[INNER_DIM][BLOCK_SIZE_X];
    __shared__ T v_j[BLOCK_SIZE_X][BLOCK_SIZE_Y];
    __shared__ T o_i[BLOCK_SIZE_Y][BLOCK_SIZE_X];
    __shared__ T answer_shmem[BLOCK_SIZE_Y][BLOCK_SIZE_X];
    __shared__ T m_i[BLOCK_SIZE_Y]; 
    __shared__ T m_tilde_ij[BLOCK_SIZE_Y]; 
    __shared__ T m_new_i[BLOCK_SIZE_Y];
    //__shared__ T l[BLOCK_SIZE_Y] = {0}; // Question, why do we even need this?
    __shared__ T l_i[BLOCK_SIZE_Y];
    __shared__ T l_tilde_ij[BLOCK_SIZE_Y];
    __shared__ T l_new_i[BLOCK_SIZE_Y]; 

    // Statistics to compute how much sparsity there truly is in the fusion.
    __shared__ float average_sparsity;
    __shared__ int num_done;

    if (tx == 0 && ty == 0) {
        average_sparsity = 0;
        num_done = 0;
    } 

    // Initialization of: l_i, m_i. TODO: initialize O_i.
    if (tx == 0 && ty+blockDim.y*blockIdx.y < seq_length) {
        l_i[ty] = l[ty+blockDim.y*blockIdx.y];
        m_i[ty] = m[ty+blockDim.y*blockIdx.y];
	    l_tilde_ij[ty] = 0;
	    l_new_i[ty] = 0;
    } else if (tx == 0) { // TODO, figure out if this is necessary. Don't think it is necessary. 
        l_i[ty] = 0;
        m_i[ty] = 0;
	    l_tilde_ij[ty] = 0;
	    l_new_i[ty] = 0;
    }

    // Load O_i. This is indepedent of the outer loop (with induction variable j in original algorithm).

    // Collaboratively load O_i.
    if (row < seq_length && col < head_hidden_dim) {
        o_i[ty][tx] = answer[idx_output(batch_num, row, head_num, col)];
    } else {
        o_i[ty][tx] = 0;
    }

    // Now, we parallelize over the inner loop, but still retain the outer loop.
    for (int j = 0; j < ceil(float(seq_length) / float(BLOCK_SIZE_X)); j++) {

        // Over here, we compute the average statistics of how sparse S_{i,j} is, delete later.
        if (blocked_is_compute(row, j*BLOCK_SIZE_X+tx, sparsity_param)) {
            atomicAdd(&num_done, 1);
            __syncthreads();
            // Then we add to the average sparsity.
            if (tx == 0 && ty == 0) {
                average_sparsity = (average_sparsity * (j+1) + (float(num_done)/float(BLOCK_SIZE_X * BLOCK_SIZE_Y)))/(j+2);
            }
        }

        // We first load V_j. 

        // Loading V_j. We transpose the x and y dimension for loading over here.
        if (tx+blockDim.x*bx < seq_length && j*BLOCK_SIZE_X+ty< hidden_dim) {
            v_j[tx][ty] = values[idx_values(batch_num, tx+blockDim.x*bx, head_num, j*BLOCK_SIZE_X+ty)];
        } else {
            v_j[tx][ty] = 0;
        }

        // Step 1: Compute S_{i,j} -> This is Q_i@K_j. We must tile this matrix multiplication.
        //          The queries will be in registers, the keys will be in shmem.
        for (int a = 0; a < ceil(float(hidden_dim) / float(BLOCK_SIZE_X)); a++) {
            // Let's first load the queries.
            __syncthreads();
            if (row < seq_length && INNER_DIM*a+tx < hidden_dim) {
                queries_shmem[ty][tx] = queries[idx_queries(batch_num, row, head_num, INNER_DIM*a+tx)];
            } else if (tx < INNER_DIM) { // This is to capture a malicious boundary case: a thread id is < INNER_DIM, but exceeds hidden_dim.
                queries_shmem[ty][tx] = 0;
            }

            // Next, we collaboratively load the keys.
            if (tx+blockDim.x*bx < seq_length && INNER_DIM*a+ty < hidden_dim) {
                keys_shmem[tx][ty] = keys[idx_keys(batch_num, tx+blockDim.x*bx, head_num, INNER_DIM*a+ty)];
            } else if (ty < INNER_DIM) { // Same as the above.
                keys_shmem[tx][ty] = 0;
            }

            // We do the matrix_multiplication here.
            for (int k = 0; k < INNER_DIM; k++) {
                answer_shmem[ty][tx] += queries_shmem[ty][k] * keys_shmem[k][tx];
            }

            __syncthreads();
        }

        // Line 10 in algorithm.
        // Over here, we compute ~m_{i,j}.
        if (tx == 0) {
            for (int k = 0; k < BLOCK_SIZE_X; k++) {
                m_tilde_ij[ty] = fmaxf(answer_shmem[ty][k], m_tilde_ij[ty]);
            }
        }

        // Over here, we compute ~P_{i,j}.
        answer_shmem[ty][tx] = expf(answer_shmem[ty][tx] - m_tilde_ij[ty]);

        // Over here, we compute ~l_{i,j}.
        if (tx == 0) {
            for (int k = 0; k < BLOCK_SIZE_X; k++) {
                l_tilde_ij[ty] += answer_shmem[ty][k];
            }
        }

        // Line 11.
        // Compute m^{new}_{i}.
        if (tx == 0) {
            m_new_i[ty] = fmaxf(m_i[ty], m_tilde_ij[ty]);
        }

        // Compute l^{new}_{i}
        if (tx == 0) {
            l_new_i[ty] = expf(m_i[ty]-m_new_i[ty])*l_i[ty] + expf(m_tilde_ij[ty]-m_new_i[ty])*l_tilde_ij[ty];
        }

        // Line 12.
        // Compute O_i -> write to HBM.
        if (row < seq_length && col < hidden_dim) {
            T temp_answer = 0;
            for (int a = 0; a < BLOCK_SIZE_X; a++) {
                temp_answer += answer_shmem[ty][a] * v_j[a][tx];
            }

            temp_answer *= expf(m_tilde_ij[ty] - m_new_i[ty]);

            temp_answer += l_i[ty]*expf(m_i[ty] - m_new_i[ty])*o_i[ty][tx];
            
            if (l_new_i[ty]) {
                temp_answer /= l_new_i[ty]; 
            }

            answer[idx_output(batch_num, row, head_num, col)] = temp_answer;
        }

        // Line 13.
        // Compute l_i <- l_i^{new} and m_i <- m_i^{new}
        if (tx == 0) {
            l_i[ty] = l_new_i[ty];
            m_i[ty] = m_new_i[ty];
        }

        // Reset the num_done counter for statistics. TODO, remove later.
        if (tx == 0 && ty == 0) {
            num_done = 0;
        }
    }

    if (tx == 0 && ty == 0 && (by == 0 || by == (gridDim.y / 2) || (by == gridDim.y - 1)) && bz == 0) {
        printf("blockIdx.x: %d, blockIdx.y: %d, sparsity: %f\n", blockIdx.x, blockIdx.y, average_sparsity);
    }
}

// Now, BLOCK_SIZE_X is b_c whilst BLOCK_SIZE_Y is b_r in the flash-attention paper:
//     https://arxiv.org/pdf/2205.14135.pdf
template<class T>
void blocked_launcher(T* queries_dev, T* keys_dev, T* values_dev, T* answer_dev, T * l_dev, T* m_dev, int batch, int num_heads, int seq_length, int hidden_dim, int sparsity_param) {

    // Now, what's the size of of O? seq_length X hidden_dim.
    // Naive spawning.
    assert(hidden_dim % num_heads == 0 && "Incorrect hidden dimension size");
    int head_hidden_dim = hidden_dim / num_heads;
    // This is an interesting way to parallelise.
    //dim3 GridSize(ceil(float(seq_length) / float(BLOCK_SIZE_X)), ceil(float(seq_length) / float(BLOCK_SIZE_Y)), batch*num_heads);
    dim3 GridSize(1, ceil(float(seq_length) / float(BLOCK_SIZE_Y)), batch*num_heads); // Let's see if this is correct.
    dim3 BlockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);

    blocked_kernel<float><<<GridSize,BlockSize>>>(queries_dev, keys_dev, values_dev, 
                                                    answer_dev, l_dev, m_dev, sparsity_param, batch, num_heads, 
                                                    seq_length, hidden_dim); 
}

int main() {

    int batch = 32; int seq_length = 1024; int num_heads = 12; int hidden_dim = 768; int sparsity_param = 128;
    int tensor_size = batch * seq_length * hidden_dim;

    float * queries = new float[tensor_size]; float * keys = new float [tensor_size]; float * values = new float[tensor_size];
    float * answer = new float[tensor_size];
    float * l = new float[seq_length]; 
    float * m = new float[seq_length];
    std::memset(l, 0, seq_length);
    std::memset(m, -1e9, seq_length);

    // Initialize random float values.
    matrix_fill(queries, tensor_size);
    matrix_fill(keys, tensor_size);
    matrix_fill(values, tensor_size);

    // GPU memory.
    float * queries_dev; float * keys_dev; float * values_dev; float * answer_dev; float * l_dev; float * m_dev;
    CHECK_CUDA(cudaMalloc(&queries_dev, sizeof(float)*tensor_size));
    CHECK_CUDA(cudaMalloc(&keys_dev, sizeof(float)*tensor_size));
    CHECK_CUDA(cudaMalloc(&values_dev, sizeof(float)*tensor_size));
    CHECK_CUDA(cudaMalloc(&answer_dev, sizeof(float)*tensor_size));
    CHECK_CUDA(cudaMalloc(&l_dev, sizeof(float)*seq_length));
    CHECK_CUDA(cudaMalloc(&m_dev, sizeof(float)*seq_length));

    // Copy tensors to GPU.
    CHECK_CUDA(cudaMemcpy(queries_dev,queries, sizeof(float)*tensor_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(keys_dev,keys, sizeof(float)*tensor_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(values_dev,values, sizeof(float)*tensor_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(answer_dev,answer, sizeof(float)*tensor_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(l_dev,l, sizeof(float)*seq_length, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(m_dev,m, sizeof(float)*seq_length, cudaMemcpyHostToDevice));

    cudaDeviceSynchronize();
    auto time_start = time_now();
    blocked_launcher(queries_dev, keys_dev, values_dev, 
                        answer_dev,l_dev,m_dev, batch, num_heads, seq_length, 
                            hidden_dim, sparsity_param);
    cudaDeviceSynchronize();
    auto time_elapsed = time_elapsed_us(time_start);
    std::cout << "Time elapsed: " << time_elapsed << std::endl;
    auto err = cudaGetLastError();
    std::cout << cudaGetErrorString(err) << std::endl;

    return 0;
}
