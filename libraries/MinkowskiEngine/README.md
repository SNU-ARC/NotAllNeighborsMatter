In `convolution_kernel.cu`,

DEFAULT: minkowski custom kernel + cublas

COPY_GEMM: only cublas

DIRECT_GEMM: only minkowski custom kernel

Change mode at `model = MinkUNet34C(3, 20, CONV_MODE="COPY_GEMM").to(device)` in `examples/indoor.py`
