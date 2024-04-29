#include "blocked.h"
#include <cmath>
#include "../../utils.h"

#define BLOCK_SIZE_X 16 // This is b_c. -> this is also our coarsening factor.
#define BLOCK_SIZE_Y 16 // This is b_r.
#define INNER_DIM 8 // This is the inner dim we use in all the mat-muls. 
// Ensure that INNER_DIM is always smaller than Min(BLOCK_SIZE_X, BLOCK_SIZE_Y).

// Queries -> row-major, Keys^T -> row-major, Values -> row-major. [batch, num_heads, seq_length, hidden_dim].
template<class T>
__global__ void blocked(T* queries, T* keys, T* values, T* answer, T * l, T * m, int sparsity_param, int batch, 
                            int num_heads, int seq_length, int hidden_dim) {

    #define idx_queries(b,s,n,h) (((b)*num_heads+(s))*seq_length+(n))*hidden_dim+(h)
    #define idx_keys(b,s,n,h) (((b)*num_heads+(s))*seq_length+(n))*hidden_dim+(h)
    #define idx_values(b,s,n,h) (((b)*num_heads+(s))*seq_length+(n))*hidden_dim+(h)
    #define idx_output(i,j,k,l) (((i)*num_heads+(j))*seq_length+(k))*seq_length+(l)

    int batch = blockIdx.z / batch;
    int head = blockIdx.z % batch;

    int tx = threadIdx.x; int ty = threadIdx.y;
    int bx = blockIdx.x; int by = blockIdx.y;
    int row = by * gridDim.y + ty; int col = bx * gridDim.x + tx;

    // THis is for the first inner loop, computing the QiKj product.
    __shared__ T queries_shmem[BLOCK_SIZE_Y][INNER_DIM];
    __shared__ T keys_shmem[INNER_DIM][BLOCK_SIZE_X];
    __shared__ T v_j[BLOCK_SIZE_X][BLOCK_SIZE_Y];
    __shared__ T o_i[BLOCK_SIZE_X][BLOCK_SIZE_Y];
    __shared__ T answer_shmem[BLOCK_SIZE_Y][BLOCK_SIZE_X];
    __shared__ T m_i[BLOCK_SIZE_Y]; // TODO, initialize this to -inf.
    __shared__ T m_tilde_ij[BLOCK_SIZE_Y]; // TODO, initialize this to -inf.
    __shared__ T m_new_i[BLOCK_SIZE_Y];
    __shared__ T l[BLOCK_SIZE_Y] = {0};
    __shared__ T l_i[BLOCK_SIZE_Y] = {0};
    __shared__ T l_tilde_ij[BLOCK_SIZE_Y] = {0};
    __shared__ T l_new_i[BLOCK_SIZE_Y] = {0};

    // Initialization of: l_i, m_i. TODO: initialize O_i.
    if (tx == 0 && ty+blockDim.y*blockIdx.y < seq_length) {
        l_i[ty] = l[ty+blockDim.y*blockIdx.y];
        m_i[ty] = m[ty+blockDim.y*blockIdx.y];
    } else if (tx == 0) { // TODO, figure out if this is necessary. Don't think it is necessary. 
        l_i[ty] = 0;
        m_i[ty] = 0;
    }

    // Now, we parallelize over the inner loop, but still retain the outer loop.
    for (int j = 0; j < ceil(float(seq_length) / float(BLOCK_SIZE_X)); j++) {

        // We first load V_j and O_i.

        // Loading V_j.

        // Step 1: Compute S_{i,j} -> This is Q_i@K_j. We must tile this matrix multiplication.
        //          The queries will be in registers, the keys will be in shmem.
        for (int a = 0; a < ceil(float(hidden_dim) / float(BLOCK_SIZE_X)); a++) {
            // Let's first load the queries.
            __syncthreads();
            if (row < seq_length && INNER_DIM*a+tx < hidden_dim && tx < INNER_DIM) {
                queries_shmem[ty][tx] = queries[idx_queries(batch, row, head, INNER_DIM*a+tx)];
            } else if (tx < INNER_DIM) {
                queries_shmem[ty][tx] = 0;
            }

            // Next, we collaboratively load the keys.
            if (col < seq_length && INNER_DIM*a+ty < hidden_dim) {
                keys_shmem[ty][tx] = keys[idx_keys(batch, col, head, INNER_DIM*a+ty)];
            } else if (ty < INNER_DIM) {
                keys_shmem[ty][tx] = 0;
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
        if (row < seq_length && col < seq_length) {
            T temp_answer = 0;
            for (int a = 0; a < BLOCK_SIZE_X; a++) {
                temp_answer += answer_shmem[ty][a] * v_j[a][tx];
            }

            temp_answer *= expf(m_tilde_ij[ty] - m_new_i[ty]);

            temp_answer += l_i[ty]*expf(m_i[ty] - m_new_i[ty])o_i[ty][tx];
            
            temp_answer /= l_new_i[ty]; // TODO, maybe division by 0.

            answer[idx_output(batch, head, row, col)] = temp_answer;
        }

        // Line 13.
        // Compute l_i <- l_i^{new} and m_i <- m_i^{new}
        if (tx == 0) {
            l_i[ty] = l_new_i[ty]
            m_i[ty] = m_new_i[ty]
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