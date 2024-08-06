add_executable(nccl_overlap main.cc compute.hpp communicate.hpp)
target_link_libraries(nccl_overlap PUBLIC CUDA::cublas ${MPI_CXX_LIBRARIES} ${NCCL_LIBRARIES})
# set -fPIC globally