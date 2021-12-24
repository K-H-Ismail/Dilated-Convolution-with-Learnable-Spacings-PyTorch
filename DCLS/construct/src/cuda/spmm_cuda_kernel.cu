#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <curand.h>
#include <curand_kernel.h>
#include <cusparse.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <assert.h>

#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparseLt.h>       // cusparseLt header

using namespace std;

#define ERR_NE(X,Y) do { if ((X) != (Y)) { \
    fprintf(stderr,"Error in %s at %s:%d exit-status:%d\n",__func__,__FILE__,__LINE__,X); \
    exit(-1);}} while(0)
#define CUDA_CALL(X) ERR_NE((X),cudaSuccess)
#define CUSPARSE_CALL(X) ERR_NE((X),CUSPARSE_STATUS_SUCCESS)

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
    }                                                                          \
}                                                                              \

template<class T>
struct reCuBuffer
{
    T* data = NULL;
    int len = 0;
};

template<class T>
void resize(reCuBuffer<T>& buffer, int size)
{
    if(size > buffer.len)
    {
        if(buffer.len > 0)
            CUDA_CALL(cudaFree(buffer.data));
            
        CUDA_CALL(cudaMalloc( &(buffer.data), size));
        buffer.len = size;
    }
}

#define num_device 16

static reCuBuffer<int>   nnzPerCol_[num_device], ColInd_[num_device], RowPtr_[num_device];
static reCuBuffer<float> csrVal_[num_device], tranBuffer_[num_device];
static reCuBuffer<void>  dBuffer_[num_device];

struct cublasHandle_
{
    cublasHandle_t handle_;
    bool init = false;
};
static cublasHandle_ handle2_[num_device];


void sparse_mm_dense_cusparse_backend(const int & cuda_device_id, const int & m, const int & n, const int & p, float * dA, float * dB, float * dC)
{
    assert(cuda_device_id>=0);
    cudaSetDevice(cuda_device_id);

    reCuBuffer<int>& nnzPerCol    = nnzPerCol_[cuda_device_id];
    reCuBuffer<int>& ColInd       = ColInd_[cuda_device_id];
    reCuBuffer<int>& RowPtr       = RowPtr_[cuda_device_id];
    reCuBuffer<float>& csrVal     = csrVal_[cuda_device_id];

    int total_nnz;
    resize(nnzPerCol, m * sizeof(int));
    
    cusparseHandle_t  handle;
    CUSPARSE_CALL(cusparseCreate(&handle));
    
#if __CUDACC_VER_MAJOR__ == 10

    // transform dense A to csr
    cusparseMatDescr_t descrX;
    CUSPARSE_CALL(cusparseCreateMatDescr(&descrX));

    CUSPARSE_CALL(cusparseSnnz(handle, CUSPARSE_DIRECTION_COLUMN, n, m, descrX, dA, n, nnzPerCol.data, &total_nnz));
    
    resize(csrVal, total_nnz * sizeof(float));
    resize(ColInd, total_nnz * sizeof(int));
    resize(RowPtr, (m+1) * sizeof(int));  
    
    CUSPARSE_CALL(cusparseSdense2csc(handle, n, m, descrX, dA, n, nnzPerCol.data, csrVal.data, ColInd.data, RowPtr.data));
    
    reCuBuffer<float>& tranBuffer = tranBuffer_[cuda_device_id];

    // CT = A * BT
    resize(tranBuffer, m * p * sizeof(float));

    // B * C
    cusparseMatDescr_t descrA;
    CUSPARSE_CALL(cusparseCreateMatDescr(&descrA));
    CUSPARSE_CALL(cusparseSetMatType(descrA,CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CALL(cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO));

    float alpha = 1.0f;
    float beta  = 0.0f;
    CUSPARSE_CALL(cusparseScsrmm2(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_TRANSPOSE,
                  m,p,n,total_nnz,&alpha,descrA,csrVal.data,RowPtr.data, ColInd.data,dB,p,&beta,tranBuffer.data,m));
    CUSPARSE_CALL(cusparseDestroyMatDescr(descrA));

    // cublasDestroy will synchronize the device
    cublasHandle_t& handle2 = handle2_[cuda_device_id].handle_;
    if(!handle2_[cuda_device_id].init)
    {
        cublasCreate(&handle2);
        handle2_[cuda_device_id].init = true;
    }

    // C need TRANSPOSE
    cublasSgeam(handle2, CUBLAS_OP_T, CUBLAS_OP_T, p, m, &alpha, tranBuffer.data, m, &beta, tranBuffer.data, m, dC, p);
    //cublasDestroy(handle2); 
    CUSPARSE_CALL(cusparseDestroyMatDescr(descrX));     
#endif

#if __CUDACC_VER_MAJOR__ == 11
    
    reCuBuffer<void>& dBuffer = dBuffer_[cuda_device_id];

    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t descrX, matB, matC;
    
    size_t bufferSize = 0;
    
    // Create dense matrix descrX   
    CUSPARSE_CALL(cusparseCreateDnMat(&descrX, m, n, n, dA, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    
    // Create sparse matrix A in CSR format    
    resize(RowPtr, (m+1) * sizeof(int));      
    CUSPARSE_CALL(cusparseCreateCsr(&matA, m, n, 0, RowPtr.data, NULL, NULL,
                                    CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    
    // allocate an external buffer if needed    
    CUSPARSE_CALL(cusparseDenseToSparse_bufferSize(handle, descrX, matA, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,&bufferSize));    
    resize(dBuffer, bufferSize);     
 
    
    // analyze Sparse to Dense conversion    
    CUSPARSE_CALL(cusparseDenseToSparse_analysis(handle, descrX, matA, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer.data));
    
    // get number of non-zero elements
    int64_t num_rows_tmp, num_cols_tmp, nnz;    
    CUSPARSE_CALL(cusparseSpMatGetSize(matA, &num_rows_tmp, &num_cols_tmp, &nnz));
    
    // resize CSR column indices and values    
    resize(csrVal, nnz * sizeof(float));
    resize(ColInd, nnz * sizeof(int));
   
    
    // reset offsets, column indices, and values pointers
    CUSPARSE_CALL(cusparseCsrSetPointers(matA, RowPtr.data, ColInd.data, csrVal.data));
    
    // execute Sparse to Dense conversion
    CUSPARSE_CALL(cusparseDenseToSparse_convert(handle, descrX, matA, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer.data));    
    
    // Create dense matrix B
    int ldb = p;
    CUSPARSE_CALL(cusparseCreateDnMat(&matB, n, p, ldb, dB, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    // Create dense matrix C
    int ldc = p;
    CUSPARSE_CALL(cusparseCreateDnMat(&matC, m, p, ldc, dC, CUDA_R_32F, CUSPARSE_ORDER_ROW));

    // allocate an external buffer if needed
    float alpha = 1.0f;
    float beta  = 0.0f;

    CUSPARSE_CALL(cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &alpha, matA, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, &bufferSize));
    resize(dBuffer, bufferSize);

    // execute SpMM
    CUSPARSE_CALL(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, matB, &beta, matC, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, dBuffer.data));

    // destroy matrix/vector descriptors
    CUSPARSE_CALL(cusparseDestroyDnMat(descrX));    
    CUSPARSE_CALL(cusparseDestroySpMat(matA));
    CUSPARSE_CALL(cusparseDestroyDnMat(matB));
    CUSPARSE_CALL(cusparseDestroyDnMat(matC));
#endif

    CUSPARSE_CALL(cusparseDestroy(handle));    
}


void sparse_mm_dense_cusparse_backend_lt(const int & cuda_device_id, const int & m, const int & n, const int & k, float * dA, float * dB, float * dC)
{
    int major_cc, minor_cc;
    CHECK_CUDA( cudaDeviceGetAttribute(&major_cc,
                                       cudaDevAttrComputeCapabilityMajor, 0) )
    CHECK_CUDA( cudaDeviceGetAttribute(&minor_cc,
                                       cudaDevAttrComputeCapabilityMinor, 0) )
    if (!(major_cc == 8 && minor_cc == 0) &&
        !(major_cc == 8 && minor_cc == 6)) {
        std::printf("\ncusparseLt is supported only on GPU devices with"
                    " compute capability == 8.0, 8.6 current: %d.%d\n\n",
                     major_cc, minor_cc);
    }
    
    // Host problem definition, row-major order
    auto          order = CUSPARSE_ORDER_ROW;
    auto          opA   = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto          opB   = CUSPARSE_OPERATION_NON_TRANSPOSE;
    auto          type  = CUDA_R_32F;
    auto          compute_type = CUSPARSE_COMPUTE_TF32;

    bool     is_rowmajor    = (order == CUSPARSE_ORDER_ROW);
    bool     isA_transposed = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
    bool     isB_transposed = (opB != CUSPARSE_OPERATION_NON_TRANSPOSE);
    auto     num_A_rows     = (isA_transposed) ? k : m;
    auto     num_A_cols     = (isA_transposed) ? m : k;
    auto     num_B_rows     = (isB_transposed) ? n : k;
    auto     num_B_cols     = (isB_transposed) ? k : n;
    auto     num_C_rows     = m;
    auto     num_C_cols     = n;
    unsigned alignment      = 32;
    auto     lda            = (is_rowmajor) ? num_A_cols : num_A_rows;
    auto     ldb            = (is_rowmajor) ? num_B_cols : num_B_rows;
    auto     ldc            = (is_rowmajor) ? num_C_cols : num_C_rows;



    float alpha = 1.0f;
    float beta  = 0.0f;    
    //--------------------------------------------------------------------------
    // Device memory management
    float *dD;
    float *dA_compressed;
    int    *d_valid;

    CHECK_CUDA( cudaMalloc((void**) &d_valid, sizeof(d_valid)) )
    dD = dC;

    //--------------------------------------------------------------------------
    cusparseLtHandle_t             handle;
    cusparseLtMatDescriptor_t      matA, matB, matC;
    cusparseLtMatmulDescriptor_t   matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t         plan;
    cudaStream_t                   stream = nullptr;
    CHECK_CUSPARSE( cusparseLtInit(&handle) )
    // matrix descriptor initialization
    CHECK_CUSPARSE( cusparseLtStructuredDescriptorInit(
                                            &handle, &matA, num_A_rows,
                                            num_A_cols, lda, alignment,
                                            type, order,
                                            CUSPARSELT_SPARSITY_50_PERCENT) )
    CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
                                            &handle, &matB, num_B_rows,
                                            num_B_cols, ldb, alignment,
                                            type, order) )
    CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
                                            &handle, &matC, num_C_rows,
                                            num_C_cols, ldc, alignment,
                                            type, order) )
    // matmul, algorithm selection, and plan initialization
    CHECK_CUSPARSE( cusparseLtMatmulDescriptorInit(
                                            &handle, &matmul, opA, opB,
                                            &matA, &matB, &matC, &matC,
                                            compute_type) )
    CHECK_CUSPARSE( cusparseLtMatmulAlgSelectionInit(
                                            &handle, &alg_sel, &matmul,
                                            CUSPARSELT_MATMUL_ALG_DEFAULT) )
    int alg = 0;
    CHECK_CUSPARSE( cusparseLtMatmulAlgSetAttribute(
                                            &handle, &alg_sel,
                                            CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                            &alg, sizeof(alg)))
    size_t workspace_size, compressed_size;
    CHECK_CUSPARSE( cusparseLtMatmulGetWorkspace(&handle, &alg_sel,
                                                 &workspace_size))

    CHECK_CUSPARSE( cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel,
                                             workspace_size) )

    //--------------------------------------------------------------------------
    // Compress the A matrix
    CHECK_CUSPARSE( cusparseLtSpMMACompressedSize(&handle, &plan,
                                                  &compressed_size) )
    CHECK_CUDA( cudaMalloc((void**) &dA_compressed, compressed_size) )

    CHECK_CUSPARSE( cusparseLtSpMMACompress(&handle, &plan, dA,
                                            dA_compressed, stream) )
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Perform the matrix multiplication
    void*         d_workspace = nullptr;
    int           num_streams = 0;
    cudaStream_t* streams     = nullptr;
    CHECK_CUSPARSE( cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB,
                                     &beta, dC, dD, d_workspace, streams,
                                     num_streams) )
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // destroy plan and handle
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matA) )
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matB) )
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matC) )
    CHECK_CUSPARSE( cusparseLtMatmulPlanDestroy(&plan) )
    CHECK_CUSPARSE( cusparseLtDestroy(&handle) )
        
    //--------------------------------------------------------------------------
    // device memory deallocation
    /*CHECK_CUDA( cudaFree(dA_compressed) )
    CHECK_CUDA( cudaFree(dA) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )
    CHECK_CUDA( cudaFree(d_valid) )  */      
         
}