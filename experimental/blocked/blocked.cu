#include "blocked.h"
#include <cmath>
#include "../../utils.h"

#define BLOCK_SIZE_X 16 // This is b_c. -> this is also our coarsening factor.
#define BLOCK_SIZE_Y 128 // This is b_r.
#define TILE_DIM 8 // This should be BLOCK_SIZE_Y / BLOCK_SIZE_X as per regular joint-register shared memory tiling.

// Queries -> row-major, Keys^T -> row-major, Values -> row-major. [batch, num_heads, seq_length, hidden_dim].
template<class T>
__global__ void blocked(T* queries, T* keys, T* values, T* answer, int sparsity_param, int batch, 
                            int num_heads, int seq_length, int hidden_dim) {

    #define idx_queries(b,s,n,h) (((b)*num_heads+(s))*seq_length+(n))*hidden_dim+(h)
    #define idx_keys(b,s,n,h) (((b)*num_heads+(s))*seq_length+(n))*hidden_dim+(h)
    #define idx_values(b,s,n,h) (((b)*num_heads+(s))*seq_length+(n))*hidden_dim+(h)

    int batch = blockIdx.z / batch;
    int head = blockIdx.z % batch;

    // This is the query index the current thread will load.
    int row_y = blockIdx.y * blockDim.y + threadIdx.y;

    // THis is for the first inner loop, computing the QiKj product.
    T queries_regs[TILE_DIM];
    T qk_answer_regs[TILE_DIM];
    __shared__ T keys_shmem[TILE_DIM][BLOCK_SIZE_X];

    // Now, we parallelize over the inner loop, but still retain the outer loop.
    for (int j = 0; j < ceil(float(seq_length) / float(TILE_DIM)); j++) {

        // Step 1: Compute S_{i,j} -> This is Q_i@K_j. We must tile this matrix multiplication.
        //          The queries will be in registers, the keys will be in shmem.

        for (int i = 0; i < ceil(float(hidden_dim) / float(BLOCK_SIZE_X)); i++) {

            // Let's first load the queries.
            for (int coarsen = 0; coarsen < BLOCK_SIZE_X; coarsen++) {
                queries_regs[coarsen] = queries_regs[idx_queries(batch, row_y, head, j*TILE_DIM+coarsen)];
            }

            __syncthreads();
            
            // Next, we collaboratively load the keys.
            int row_shmem = threadIdx.x / TILE_DIM;
            int col_shmem = threadIdx.x % TILE_DIM; 
            if (&& row_shmem+j*TILE_DIM < hidden_dim) {
                keys_shmem[row_shmem][col_shmem] = keys[idx_keys(batch, col_shmem+, 
                                                                    head, row_shmem+j*TILE_DIM)]; // This looks a little incorrect.
            } else {
                keys_shmem[row_shmem][col_shmem] = 0;
            }

            __syncthreads();

            // We do the matrix_multiplication here.


            // Write to the answers here.
            for (int coarsen=0; coarsen < ) {

            }

        }

    }
}



// Now, BLOCK_SIZE_X is b_c whilst BLOCK_SIZE_Y is b_r in the flash-attention paper:
//     https://arxiv.org/pdf/2205.14135.pdf
template<class T>
void blocked_launcher(T* queries_dev, T* keys_dev, T* values_dev, T* answer_dev, 
                        int batch, int num_heads, int seq_length, int hidden_dim, int sparsity_param) {

    // Now, what's the size of of O? seq_length X hidden_dim.
    // Naive spawning.
    assert(hidden_dim % num_heads == 0 && "Incorrect hidden dimension size");
    int head_hidden_dim = hidden_dim / num_heads;
    Dim3 GridSize(ceil(float(head_hidden_dim) / float(BLOCK_SIZE_X)), ceil(float(seq_length) / float(BLOCK_SIZE_Y)), batch*num_heads);
    Dim3 BlockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);

    blocked<float><<<GridSize,BlockSize>>>blocked(queries_dev, keys_dev, values_dev, 
                                                    answer_dev, sparsity_param, batch, num_heads, 
                                                    seq_length, hidden_dim); 
}

int main() {

    int batch = 1; int seq_length = 10; int num_heads = 1; int hidden_dim = 10; int sparsity_param = 5;
    int tensor_size = batch * seq_length * hidden_dim;

    float * queries = new float[tensor_size]; float * keys = new float [tensor_size]; float * values = new float[tensor_size];
    float * answer = new float[tensor_size];

    // Initialize random float values.
    matrix_fill(queries, tensor_size);
    matrix_fill(keys, tensor_size);
    matrix_fill(values, tensor_size);

    // GPU memory.
    float * queries_dev; float * keys_dev; float * values_dev; float * answer_dev;
    cudaMalloc(&queries_dev, sizeof(float)*tensor_size);
    cudaMalloc(&keys_dev, sizeof(float)*tensor_size);
    cudaMalloc(&values_dev, sizeof(float)*tensor_size);
    cudaMalloc(&answer_dev, sizeof(float)*tensor_size);

    // Copy tensors to GPU.
    cudaMemcpy(queries_dev,queries, sizeof(float)*tensor_size, cudaMemcpyHostToDevice);
    cudaMemcpy(keys_dev,keys, sizeof(float)*tensor_size, cudaMemcpyHostToDevice);
    cudaMemcpy(values_dev,values, sizeof(float)*tensor_size, cudaMemcpyHostToDevice);

    blocked_launcher(queries_dev, keys_dev, values_dev, 
                        answer_dev, batch, num_heads, seq_length, 
                            hidden_dim, sparsity_param);

    return 0;
}