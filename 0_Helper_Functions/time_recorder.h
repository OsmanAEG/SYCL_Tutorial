#include <chrono>
#include <iostream>

class TimeRecorder{
public:
  TimeRecorder() : last_registered_time(0.0){}

  // start the timer
  void start(){
    start_time = std::chrono::high_resolution_clock::now();
  }

  // end the timer
  void end(){
    end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    last_registered_time = elapsed.count();
  }

  // print the last registered time
  void print_last_registered_time(){
    std::cout << "Time: " << last_registered_time << " seconds" << std::endl;
  }

private:
  std::chrono::high_resolution_clock::time_point start_time;
  std::chrono::high_resolution_clock::time_point end_time;
  double last_registered_time;
};