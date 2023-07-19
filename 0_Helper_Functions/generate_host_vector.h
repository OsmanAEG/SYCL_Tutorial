#include <vector>

template<typename Scalar_type, typename Int_type>
auto generate_filled_host_vector(Int_type N){
  std::vector<Scalar_type> host_vector(N);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<Scalar_type> dis(0.0, 100.0);

  auto rng = [&]() {
    return dis(gen);
  };

  std::generate(host_vector.begin(), host_vector.end(), rng);
  return host_vector;
}

template<typename Scalar_type, typename Int_type>
auto generate_empty_host_vector(Int_type N){
  std::vector<Scalar_type> host_vector(N);
  std::fill(host_vector.begin(), host_vector.end(), 0.0);
  return host_vector;
}