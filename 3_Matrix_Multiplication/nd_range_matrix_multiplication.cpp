#include <CL/sycl.hpp>
#include <assert.h>
#include <vector>
#include <random>
#include <algorithm>

#include "../1_Queue/print_device_information.h"
#include "../1_Queue/device_inquiry.h"
#include "../0_Helper_Functions/generate_host_vector.h"
#include "../0_Helper_Functions/time_recorder.h"
#include "../0_Helper_Functions/verify.h"

// implementation of an nd_range matrix multiplication in SYCL
template<typename Queue_T, typename Scalar_T, typename Int_T, typename Size_T>
auto nd_range_matrix_multiplication(Queue_T Q,
                                    Scalar_T* A,
                                    Scalar_T* B,
                                    Scalar_T* C,
                                    Int_T& M,
                                    Int_T& N,
                                    Int_T& P,
                                    Size_T b){
  auto event = Q.submit([&](sycl::handler& h){
    // global range and local work group size
    auto global = sycl::range<2>(M, P);
    auto local = sycl::range<2>(b, b);
    h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it){
      const int i = it.get_global_id(0);
      const int j = it.get_global_id(1);

      double c_ij = 0.0;

      for(int k = 0; k < N; ++k){
        c_ij += A[i*N + k] * B[k*P + j];
      }

      C[i*P + j] = c_ij;
    });
  });

  return event;
}

int main(){
  // create time recorder
  auto timer = TimeRecorder();

  // select queue based on platform and device number
  auto Q = get_queue(0, 0);
  print_device_information(Q);

  // create vectors on host (CPU)
  size_t M = 1 << 12;
  size_t N = 1 << 12;
  size_t P = 1 << 12;

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

  std::cout << "nd_range matrix multiplication" << std::endl;
  std::cout << "---------------------------" << std::endl;

  ////////////////////////////////////////////////////////////////
  std::cout << "Test 1: Work Group Size = 4" << std::endl;
  timer.start();
  auto event = nd_range_matrix_multiplication(Q, A_device, B_device, C_device,
                                              M, N, P, 4);
  event.wait();
  timer.end();
  timer.print_last_registered_time();
  std::cout << "---------------------------" << std::endl;

  ////////////////////////////////////////////////////////////////
  std::cout << "Test 2: Work Group Size = 8" << std::endl;
  timer.start();
  event = nd_range_matrix_multiplication(Q, A_device, B_device, C_device,
                                              M, N, P, 8);
  event.wait();
  timer.end();
  timer.print_last_registered_time();
  std::cout << "---------------------------" << std::endl;

  ////////////////////////////////////////////////////////////////
  std::cout << "Test 3: Work Group Size = 16" << std::endl;
  timer.start();
  event = nd_range_matrix_multiplication(Q, A_device, B_device, C_device,
                                              M, N, P, 16);
  event.wait();
  timer.end();
  timer.print_last_registered_time();
  std::cout << "---------------------------" << std::endl;

  ////////////////////////////////////////////////////////////////
  std::cout << "Test 4: Work Group Size = 32" << std::endl;
  timer.start();
  event = nd_range_matrix_multiplication(Q, A_device, B_device, C_device,
                                              M, N, P, 32);
  event.wait();
  timer.end();
  timer.print_last_registered_time();
  std::cout << "---------------------------" << std::endl;

  // copying data from device to host
  /*Q.memcpy(&C_host[0], C_device, M*P*sizeof(double)).wait();

  // checking the results
  verify_matrix_multiplication(A_host, B_host, C_host, M, N, P);

  std::cout << "The matrix multiplication was successful!" << std::endl;*/
}