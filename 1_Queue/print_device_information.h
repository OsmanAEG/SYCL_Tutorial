#include <CL/sycl.hpp>
#include <iostream>

// print device information
template<typename Queue_type>
void print_device_information(Queue_type& Q){
  std::cout << "-------------- DEVICE INFORMATION --------------\n"
            << "NAME: "
            << Q.get_device().template get_info<sycl::info::device::name>()
            << "\nVENDOR: "
            << Q.get_device().template get_info<sycl::info::device::vendor>()
            << "\nMAX WORKGROUP SIZE: "
            << Q.get_device().template get_info<sycl::info::device::max_work_group_size>()
            << "\n" << "------------------------------------------------\n"
            << std::endl;
}