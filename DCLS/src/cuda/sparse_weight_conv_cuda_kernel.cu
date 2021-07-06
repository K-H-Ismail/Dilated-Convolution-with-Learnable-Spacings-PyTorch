#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSparseToDense

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <math.h>
#include <vector>

#define CUDA_KERNEL_LOOP(i, n)                                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);                 \
       i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
const int kMaxGridNum = 65535;
inline int GET_BLOCKS(const int N) {
  return std::min(kMaxGridNum, (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
}

/*template <typename scalar_t>
__global__ void change_elem(scalar_t *arr, long idx, scalar_t val) {
    arr[idx] = val;
}
long to_csr(
    torch::Tensor mat,
    float* d_csr_values,
    long* d_csr_columns,
    long* d_csr_offsets) 
{   
    long num_rows = mat.size(0);
    long num_cols = mat.size(1);
    long nnz = 0;

    

    
    change_elem<long><<<1,1>>>(d_csr_offsets,0,0); // 0-base indexing
    for (long i = 0; i < num_rows; i++)
    {
        for (long j = 0; j < num_cols; j++)
        {
            float current_value = mat[i][j].item<float>();
            if (fabs(current_value) > 1e-10)
            {
                change_elem<float><<<1,1>>>(d_csr_values,nnz,current_value);
                change_elem<long><<<1,1>>>(d_csr_columns,nnz,j);                
                nnz ++;
            }
        }
        change_elem<long><<<1,1>>>(d_csr_offsets,i+1,nnz);        
    }
    return nnz;    
}*/
/*__device__ unsigned int nnz = 0;
__global__ void to_csr_kernel(
    const int n,    
    torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> mat,
    const int mat_num_rows,
    const int mat_num_cols,    
    float* dA_values,
    long* dA_columns,
    long* dA_csrOffsets) 
{

  CUDA_KERNEL_LOOP(index, n) {    
      
    int col = index % mat_num_cols;
    int row = (index / mat_num_cols) % mat_num_rows;
    
    float current_val = mat[row][col];
      
    if(fabs(current_val) > 1e-10) {      
        dA_values[nnz] = current_val; 
        dA_columns[nnz] = static_cast<long>(col);         
        atomicInc(&nnz,10);     
    }



  }
}*/

/*long to_csr_kernel(
    torch::Tensor mat,
    long *d_csr_offsets,
    long *d_csr_columns,
    float *d_csr_values) 
{   
    // Host problem definition
    int   num_rows   = mat.size(0);
    int   num_cols   = mat.size(1);
    int   ld         = num_cols;
    int   dense_size = ld * num_rows;
    
    //--------------------------------------------------------------------------
    // Device memory management
    float *d_dense;
    cudaMalloc((void**) &d_dense, dense_size * sizeof(float));
    d_dense = mat.data<float>();
        
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matB;
    cusparseDnMatDescr_t matA;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    cusparseCreate(&handle);
    
    // Create dense matrix A
    cusparseCreateDnMat(&matA, num_rows, num_cols, ld, d_dense,
                                        CUDA_R_32F, CUSPARSE_ORDER_ROW);
    // Create sparse matrix B in CSR format
    cusparseCreateCsr(&matB, num_rows, num_cols, 0,
                                      d_csr_offsets, NULL, NULL,
                                      CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    // allocate an external buffer if needed
    cusparseDenseToSparse_bufferSize(handle, matA, matB,
                                     CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                     &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    // execute Sparse to Dense conversion
    cusparseDenseToSparse_analysis(handle, matA, matB,
                                   CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                   dBuffer);
    // get number of non-zero elements
    int64_t num_rows_tmp, num_cols_tmp, nnz;
    cusparseSpMatGetSize(matB, &num_rows_tmp, &num_cols_tmp, &nnz);

    // reset offsets, column indices, and values pointers
    cusparseCsrSetPointers(matB, d_csr_offsets, d_csr_columns, d_csr_values);
    
    // execute Sparse to Dense conversion
    cusparseDenseToSparse_convert(handle, matA, matB,
                                  CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
                                  dBuffer);
    
    // destroy matrix/vector descriptors
    cusparseDestroyDnMat(matA);
    cusparseDestroySpMat(matB);
    cusparseDestroy(handle);

    //--------------------------------------------------------------------------
    // device memory deallocation
    cudaFree(dBuffer);
    cudaFree(d_dense);
    return nnz;
}*/


    
torch::Tensor sparse_mm(
    torch::Tensor mA,    
    torch::Tensor mB) 
{



    // Host problem definition
    int   A_num_rows      = mA.size(0);
    int   A_num_cols      = mA.size(1);
    int   B_num_rows      = A_num_cols;
    int   B_num_cols      = mB.size(1);  
    int   ldb             = B_num_cols;
    int   ldc             = B_num_cols;
    int   B_size          = ldb * B_num_rows;
    int   C_size          = ldc * A_num_rows;
    
    auto mA_sp = mA.to_sparse();
    int64_t num_nnz = mA_sp._nnz();
    int64_t*  dA_rows;
    int64_t*  dA_columns;
    float* dA_values;
    cudaMalloc((void**) &dA_rows, num_nnz * sizeof(int64_t)); 
    cudaMalloc((void**) &dA_columns, num_nnz * sizeof(int64_t));
    cudaMalloc((void**) &dA_values,  num_nnz * sizeof(float));
    cudaMemcpy(dA_rows, mA_sp.indices().select(0,0).data<int64_t>(), num_nnz * sizeof(int64_t), cudaMemcpyDeviceToDevice); 
    cudaMemcpy(dA_columns, mA_sp.indices().select(0,1).data<int64_t>(), num_nnz * sizeof(int64_t), cudaMemcpyDeviceToDevice); 
    cudaMemcpy(dA_values, mA_sp.values().data<float>(), num_nnz * sizeof(float), cudaMemcpyDeviceToDevice); 
    int64_t   A_nnz          = num_nnz;
    
    float* dB; 
    float* dC;
    cudaMalloc((void**) &dB, B_size * sizeof(float)); 
    cudaMemcpy(dB, mB.data<float>(), B_size * sizeof(float), cudaMemcpyDeviceToDevice);      
    cudaMalloc((void**) &dC, C_size * sizeof(float)); 

    float alpha           = 1.0;
    float beta            = 0.0;  
    

    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    
    cusparseHandle_t handle = NULL;
    cusparseDnMatDescr_t matB, matC;
    void *dBuffer    = NULL;
    size_t bufferSize = 0;
    
    cusparseCreate(&handle);
    

    // Create dense matrix B
    cusparseCreateDnMat(&matB, B_num_rows, B_num_cols, ldb, dB,
                        CUDA_R_32F, CUSPARSE_ORDER_ROW);
    // Create dense matrix C
    cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC,
                        CUDA_R_32F, CUSPARSE_ORDER_ROW);
    
    // Convert A from a dense formatting to a CSR formatting, using the GPU
    // Create sparse matrix spA in CSR format
    
    cusparseSpMatDescr_t spmatA;

    cusparseCreateCoo(&spmatA, A_num_rows, A_num_cols, A_nnz,
                      dA_rows, dA_columns, dA_values,
                      CUSPARSE_INDEX_64I, 
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
        
    // Perform matrix-matrix multiplication with the CSR-formatted matrix A
    //C = α ∗ op ( A ) ∗ op ( B ) + β ∗ C 
    // allocate an external buffer if needed
    cusparseSpMM_bufferSize(handle,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, spmatA, matB, &beta, matC, CUDA_R_32F,
                            CUSPARSE_SPMM_COO_ALG4, &bufferSize); 
    
    cudaMalloc(&dBuffer, bufferSize);    
    
    cusparseSpMM(handle, 
                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                 CUSPARSE_OPERATION_NON_TRANSPOSE, 
                 &alpha, spmatA, matB, &beta, matC, CUDA_R_32F,
                 CUSPARSE_SPMM_COO_ALG4, dBuffer);
        
    auto options = mB.options();
    torch::Tensor res = torch::from_blob(dC, {A_num_rows,B_num_cols}, options);
    
    // destroy matrix/vector descriptors
    cusparseDestroySpMat(spmatA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cusparseDestroy(handle);
    
    cudaFree(dBuffer);
    cudaFree(dA_rows);
    cudaFree(dA_columns);
    cudaFree(dA_values);
    cudaFree(dB);
    //cudaFree(dC);    
 
    return res;
}



torch::Tensor sparse_weight_conv_cuda_forward(
    torch::Tensor input,    
    torch::Tensor weight,
    torch::Tensor bias,  
    const int dilation_h, const int dilation_w, 
    const int stride_h, const int stride_w, 
    const int padding_h, const int padding_w, 
    const int groups,
    const bool sp_mm) {
    
    const int batch = input.size(0);
    const int channels_in = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int channels_out = weight.size(0);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3); 

    
    const int height_out = (height + 2 * padding_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * padding_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    
    // prepare group weight and bias
    auto weight_g = weight.view({groups, channels_out/groups, channels_in/groups, kernel_h, kernel_w});
    auto bias_g = torch::zeros_like(bias).view({groups, channels_out/groups});
    
    auto output = torch::empty({batch, channels_out , height_out , width_out}, input.options());
    

    /*auto a = torch::zeros({4,4}, weight.options());
    a[0][0] = 1.0;a[0][1] = 0.0;a[0][2] = 2.0;a[0][3] = 3.0;
    a[1][0] = 0.0;a[1][1] = 4.0;a[1][2] = 0.0;a[1][3] = 0.0;
    a[2][0] = 5.0;a[2][1] = 0.0;a[2][2] = 6.0;a[2][3] = 7.0;
    a[3][0] = 0.0;a[3][1] = 8.0;a[3][2] = 0.0;a[3][3] = 9.0;
    auto b = torch::zeros({4,3}, weight.options());
    b[0][0] = 1.0;b[0][1] = 5.0;b[0][2] = 9.0;
    b[1][0] = 2.0;b[1][1] = 6.0;b[1][2] = 10.0;
    b[2][0] = 3.0;b[2][1] = 7.0;b[2][2] = 11.0;
    b[3][0] = 4.0;b[3][1] = 8.0;b[3][2] = 12.0;*/
    
    //at::mm(torch::eye(30000,30000,input.options()),at::rand({30000,30000},input.options()));
    auto columns = at::empty({channels_in * kernel_h * kernel_w, height_out * width_out}, input.options());
    for (int elt = 0; elt < batch; elt++) {

        auto input_n = input.select(0, elt);
        auto output_n = output.select(0, elt);
        columns = at::im2col(input_n, {kernel_h,kernel_w}, {dilation_h,dilation_w}, {padding_h,padding_w}, {stride_h,stride_w});

        auto columns_g = columns.view({groups, channels_in/groups * kernel_h * kernel_w, height_out * width_out});
        auto output_g = output_n.view({groups, channels_out/groups, height_out * width_out});
        for (int g = 0; g < groups; ++g)
        {
            auto columns_gm = columns_g.select(0, g);
            auto weight_gm = weight_g.select(0, g).view({channels_out/groups, channels_in/groups * kernel_h * kernel_w});
            
            auto output_m = bias_g.select(0, g).view({channels_out/groups,1}) + at::mm(weight_gm, columns_gm);//sparse_mm
            output_g.select(0, g) = output_m;
        }
        output.select(0, elt) = output_g.view({channels_out, height_out, width_out});
    }
    return output;

}

std::vector<torch::Tensor> sparse_weight_conv_cuda_backward(
    torch::Tensor input,    
    torch::Tensor weight,
    torch::Tensor grad_output,      
    torch::Tensor bias,   
    const int dilation_h, const int dilation_w, 
    const int stride_h, const int stride_w, 
    const int padding_h, const int padding_w, 
    const int groups,
    const bool sp_mm) {
    
    auto grad_input = torch::zeros_like(input);      
    auto grad_weight = torch::zeros_like(weight);
    auto grad_bias = torch::zeros_like(bias);
    
    const int batch = input.size(0);
    const int channels_in = input.size(1);
    const int height = input.size(2);
    const int width = input.size(3);

    const int channels_out = weight.size(0);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);    
    
    const int height_out = (height + 2 * padding_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int width_out = (width + 2 * padding_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    
    // prepare group weight and bias
    auto weight_g = weight.view({groups, channels_out/groups, channels_in/groups, kernel_h, kernel_w});
    auto grad_weight_g = grad_weight.view({groups, channels_out/groups, channels_in/groups, kernel_h, kernel_w});
    auto grad_bias_g = grad_bias.view({groups, channels_out/groups});
    auto ones = at::ones({height_out * width_out}, input.options());
    
    for (int elt = 0; elt < batch; elt++) {

        auto input_n = input.select(0, elt);
        auto grad_input_n = at::empty({channels_in, height * width}, input.options());
        auto grad_output_n = grad_output.select(0, elt);   
        auto columns = at::empty({channels_in * kernel_h * kernel_w, height_out * width_out}, input.options());


        auto grad_output_g = grad_output_n.view({groups, channels_out/groups, height_out * width_out});
        auto columns_g = columns.view({groups, channels_in/groups * kernel_h * kernel_w, height_out * width_out});


        for (int g = 0; g < groups; ++g)
        {
            auto grad_output_gm = grad_output_g.select(0, g);
            auto columns_gm = columns_g.select(0, g);
            auto weight_gm = weight_g.select(0, g).view({channels_out/groups, channels_in/groups * kernel_h * kernel_w}).t();
            columns_g.select(0, g) = at::mm(weight_gm, grad_output_gm);//sparse_mm

        }
        columns = columns_g.view({channels_in * kernel_h * kernel_w, height_out * width_out});
        grad_input_n = at::col2im(columns, {height,width}, {kernel_h, kernel_w}, {dilation_h, dilation_w}, {padding_h, padding_w}, {stride_h, stride_w});      
        grad_input.select(0, elt) = grad_input_n.view({channels_in, height, width});
        
        
        
        columns = at::im2col(input_n, {kernel_h, kernel_w}, {dilation_h, dilation_w}, {padding_h, padding_w}, {stride_h, stride_w});
        columns_g = columns.view({groups, channels_in/groups * kernel_h * kernel_w, height_out * width_out});   
        
        for (int g = 0; g < groups; ++g)
        {
            auto grad_output_gm = grad_output_g.select(0, g);
            auto columns_gm = columns_g.select(0, g).t();
            auto grad_weight_gm = grad_weight_g.select(0, g)
                .view({channels_out/groups, channels_in/groups * kernel_h * kernel_w});
            auto grad_bias_gm = grad_bias_g.select(0, g);
            grad_weight_g.select(0, g) = at::addmm(grad_weight_gm, grad_output_gm, columns_gm)
                .view_as(grad_weight_g.select(0, g));
            grad_bias_g.select(0, g) = at::addmv(grad_bias_gm, grad_output_gm, ones);
        }
    
        grad_weight += grad_weight_g.view({channels_out, channels_in/groups, kernel_h, kernel_w});        

    }
    return {grad_input,
            grad_weight,
            grad_bias};
}
