#include <CL/sycl.hpp>
#include <assert.h>
#include <vector>
#include <random>
#include <algorithm>

#include "../1_Queue/print_device_information.h"
#include "../1_Queue/device_inquiry.h"
#include "../0_Helper_Functions/generate_host_vector.h"

int main(){
  // select queue based on platform and device number
  auto Q = get_queue(0, 0);
  print_device_information(Q);

  // create vectors on host (CPU)
  size_t M = 1 << 9;
  size_t N = 1 << 8;
  size_t P = 1 << 5;

  auto A_host = generate_filled_host_vector<double>(M*N);
  auto B_host = generate_filled_host_vector<double>(N*P);
  auto C_host = generate_empty_host_vector<double>(M*P);

  // allocating memory on device (GPU)
  double *A_device = sycl::malloc_device<double>(M*N, Q);
  double *B_device = sycl::malloc_device<double>(N*P, Q);
  double *C_device = sycl::malloc_device<double>(M*P, Q);

  // copying data from host to device
  Q.memcpy(A_device, &A_host[0], M*N*sizeof(double));
  Q.memcpy(B_device, &B_host[0], N*P*sizeof(double));
  Q.wait();

  // executing the kernel
  auto event = Q.submit([&](sycl::handler& h){
    h.parallel_for(sycl::range<2>{M, P}, [=](sycl::id<2> idx){
      const int i = idx[0];
      const int j = idx[1];

      double c_ij = 0.0;

      for(int k = 0; k < N; ++k){
        c_ij += A_device[i*N + k] * B_device[k*P + j];
      }

      C_device[i*P + j] = c_ij;
    });
  });

  event.wait();

  // copying data from device to host
  Q.memcpy(&C_host[0], C_device, M*P*sizeof(double)).wait();

  // checking the results
  for(int i = 0; i < M; i++){
    for(int j = 0; j < P; j++){
      double c_ij = 0.0;

      for(int k = 0; k < N; k++){
        c_ij += A_host[i*N + k] * B_host[k*P + j];
      }

      assert(std::fabs(C_host[i*P + j] - c_ij) < 1e-6);
    }
  }

  std::cout << "The matrix multiplication was successful!" << std::endl;
}