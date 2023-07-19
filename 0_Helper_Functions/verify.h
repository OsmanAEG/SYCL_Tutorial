#include <assert.h>

// verification function for vector addition
template<typename Vector_type, typename Int_type>
void verify_vector_addition(Vector_type A,
                            Vector_type B,
                            Vector_type C,
                            Int_type& N,
                            float tolerance = 1.0e-6){
  for(int i = 0; i < N; ++i){
    assert(std::fabs(C[i] - (A[i] + B[i])) < tolerance);
  }
}

// verification function for matrix multiplication
template<typename Vector_type, typename Int_type>
void verify_matrix_multiplication(Vector_type A,
                                  Vector_type B,
                                  Vector_type C,
                                  Int_type& M,
                                  Int_type& N,
                                  Int_type& P,
                                  float tolerance = 1.0e-6){
  for(int i = 0; i < M; ++i){
    for(int j = 0; j < P; ++j){
      double c_ij = 0.0;

      for(int k = 0; k < N; ++k){
        c_ij += A[i*N + k] * B[k*P + j];
      }

      assert(std::fabs(C[i*P + j] - c_ij) < tolerance);
    }
  }
}