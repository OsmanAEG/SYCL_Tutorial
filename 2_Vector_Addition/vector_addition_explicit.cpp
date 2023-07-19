#include <CL/sycl.hpp>
#include <assert.h>
#include <vector>
#include <random>
#include <algorithm>

#include "../1_Queue/print_device_information.h"
#include "../1_Queue/device_inquiry.h"

int main(){
  // select queue based on platform and device number
  auto Q = get_queue(0, 0);
  print_device_information(Q);

  // create vectors on host (CPU)
  size_t N = 1 << 10;

  std::vector<double> A_host(N);
  std::vector<double> B_host(N);
  std::vector<double> C_host(N);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(0.0, 100.0);

  auto rng = [&]() {
    return dis(gen);
  };

  std::generate(A_host.begin(), A_host.end(), rng);
  std::generate(B_host.begin(), B_host.end(), rng);

  std::fill(C_host.begin(), C_host.end(), 0.0);

  // allocating memory on device (GPU)
  double *A_device = sycl::malloc_device<double>(N, Q);
  double *B_device = sycl::malloc_device<double>(N, Q);
  double *C_device = sycl::malloc_device<double>(N, Q);

  // copying data from host to device
  Q.memcpy(A_device, &A_host[0], N*sizeof(double));
  Q.memcpy(B_device, &B_host[0], N*sizeof(double));

  // executing the kernel
  Q.submit([&](sycl::handler& h){
    h.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx){
      C_device[idx] = A_device[idx] + B_device[idx];
    });
  });

  Q.wait(); // waiting for the kernel to finish

  // copying data from device to host
  Q.memcpy(&C_host[0], C_device, N*sizeof(double));

  // checking the results
  for(size_t i = 0; i < N; i++){
    assert(std::fabs(C_host[i] - (A_host[i] + B_host[i])) < 1e-6);
  }

  std::cout << "The vector addition was successful!" << std::endl;
}