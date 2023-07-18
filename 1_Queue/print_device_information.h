#include <CL/sycl.hpp>
#include <iostream>

// print device information
template<typename Queue_type>
void print_device_information(Queue_type& Q){
  std::cout << "DEVICE NAME: "
            << Q.get_device().template get_info<sycl::info::device::name>()
            << "\nDEVICE VENDOR: "
            << Q.get_device().template get_info<sycl::info::device::vendor>()
            << "\n" << std::endl;
}