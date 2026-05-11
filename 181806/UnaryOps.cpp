// Reference: aten/src/ATen/native/UnaryOps.cpp (signbit_out dispatch only)
// This is the structured dispatch that routes signbit to CPU or CUDA stubs.

TORCH_IMPL_FUNC(signbit_out) (const Tensor& self, const Tensor& result) {
  if (self.dtype() == at::kBool) {
    result.fill_(false);
  } else {
    signbit_stub(device_type(), *this);
  }
}
