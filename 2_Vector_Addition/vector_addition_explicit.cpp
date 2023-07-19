#include <CL/sycl.hpp>
#include <vector>
#include <random>
#include <algorithm>

#include "../1_Queue/print_device_information.h"
#include "../1_Queue/device_inquiry.h"
#include "../0_Helper_Functions/generate_host_vector.h"
#include "../0_Helper_Functions/verify.h"

int main(){
  // select queue based on platform and device number
  auto Q = get_queue(0, 0);
  print_device_information(Q);

  // create vectors on host (CPU)
  size_t N = 1 << 10;

  auto A_host = generate_filled_host_vector<double>(N);
  auto B_host = generate_filled_host_vector<double>(N);
  auto C_host = generate_empty_host_vector<double>(N);

  // allocating memory on device (GPU)
  double *A_device = sycl::malloc_device<double>(N, Q);
  double *B_device = sycl::malloc_device<double>(N, Q);
  double *C_device = sycl::malloc_device<double>(N, Q);

  // copying data from host to device
  Q.memcpy(A_device, &A_host[0], N*sizeof(double));
  Q.memcpy(B_device, &B_host[0], N*sizeof(double));
  Q.wait();

  // executing the kernel
  auto event = Q.submit([&](sycl::handler& h){
    h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx){
      C_device[idx] = A_device[idx] + B_device[idx];
    });
  });

  event.wait();

  // copying data from device to host
  Q.memcpy(&C_host[0], C_device, N*sizeof(double)).wait();

  // checking the results
  verify_vector_addition(A_host, B_host, C_host, N);

  std::cout << "The vector addition was successful!" << std::endl;
}