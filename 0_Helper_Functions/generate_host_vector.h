#include <vector>

// generate a vector on host, filled with random numbers
template<typename Scalar_T, typename Int_T>
auto generate_filled_host_vector(Int_T N){
  std::vector<Scalar_T> host_vector(N);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<Scalar_T> dis(0.0, 100.0);

  auto rng = [&]() {
    return dis(gen);
  };

  std::generate(host_vector.begin(), host_vector.end(), rng);
  return host_vector;
}

// generate a vector on host, filled with zeros
template<typename Scalar_T, typename Int_T>
auto generate_empty_host_vector(Int_T N){
  std::vector<Scalar_T> host_vector(N);
  std::fill(host_vector.begin(), host_vector.end(), 0.0);
  return host_vector;
}