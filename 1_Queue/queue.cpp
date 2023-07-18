#include <CL/sycl.hpp>

#include "print_device_information.h"
#include "device_inquiry.h"

int main(){
  // selecting a default device for queue
  /*sycl::queue Q{sycl::default_selector_v};
  print_device_information(Q);*/

  // inquiring about all available devices
  print_available_devices();
}