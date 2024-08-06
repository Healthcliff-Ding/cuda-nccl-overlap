add_executable(nccl_overlap main.cc compute.hpp communicate.hpp)
target_link_libraries(nccl_overlap PUBLIC CUDA::cublas ${MPI_CXX_LIBRARIES} ${NCCL_LIBRARIES})
set_property(TARGET nccl_overlap PROPERTY POSITION_INDEPENDENT_CODE  ON)