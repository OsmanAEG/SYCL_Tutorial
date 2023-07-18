#include <CL/sycl.hpp>

// printing all available devices on the available platforms
void print_available_devices(){
  int p = 0;
  auto platforms = sycl::platform::get_platforms();
  for(auto& platform : platforms){
    std::cout << "PLATFORM " << p << " | "
              << platform.template get_info<sycl::info::platform::name>()
              << " | WITH DEVICES:\n"
              << "---------------------------------------------------------"
              << std::endl;
    int d = 0;
    auto devices = platform.get_devices();
    for(auto& device : devices){
      // printing device information
      std::cout << " - DEVICE " << d << ": "
                << device.template get_info<sycl::info::device::name>()
                << std::endl;
      d++;
    }

    p++;
  }
}