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

// selecting a device based on platform and device number
sycl::queue get_queue(int platform_index = 0, int device_index = 0){
  // get the available platforms
  auto platforms = sycl::platform::get_platforms();

  // select the platform based on the platform index
  auto selected_platform = platforms[platform_index];

  // get the devices on the selected platform
  auto devices = selected_platform.get_devices();
  auto selected_device = devices[device_index];

  // create the queue based on the selected device
  sycl::queue q(selected_device);
  return q;
}