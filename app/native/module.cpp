// One extension module for all native kernels (built by app/native/__init__.py).
#include <torch/extension.h>

void register_lm(pybind11::module &m);
void register_vibevoice(pybind11::module &m);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    register_lm(m);
    register_vibevoice(m);
}
