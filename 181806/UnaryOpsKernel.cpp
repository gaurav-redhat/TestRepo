// CPU reference: aten/src/ATen/native/cpu/UnaryOpsKernel.cpp (signbit_kernel only)
// This is the CPU kernel that correctly handles signbit for float16/bfloat16
// by promoting to float32 (which preserves NaN sign bits on CPU).

static void signbit_kernel(TensorIteratorBase& iter){
  // NOTE: signbit does not always support integral arguments.
  AT_DISPATCH_SWITCH(iter.input_dtype(), "signbit_cpu",
      AT_DISPATCH_CASE_INTEGRAL_TYPES([&] {
        cpu_kernel(iter, [](scalar_t a) -> bool { return c10::is_negative(a); });
      })
      AT_DISPATCH_CASE_FLOATING_TYPES_AND2(kBFloat16, ScalarType::Half, [&] {
        using opmath_t = at::opmath_type<scalar_t>;
        cpu_kernel(iter, [](scalar_t a) -> bool { return std::signbit(opmath_t{a}); });
      })
    );
}
