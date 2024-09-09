from .arch_base import Arch
import math

class CGRA_VER15(Arch):
    def __init__(self):
        super().__init__()  # Call the base class constructor without arguments
        
        # After calling the base class constructor, set the properties
        self.core = "CGRA_VER15"
        self.sm_count = 16
        self.base_freq = 0.8 * 1e9
        self.max_freq = 1.0 * 1e9
        self.tensor_cores_per_sm = 1
        self.tensor_core_shape = (8,8,1)
        self.fp32_cores_per_sm = 64
        self.ddr_bandwidth = 12.8 * 1e9
        self.ddr_capacity = 2 * (1024**3)
        # self.l2_bandwidth = self.ddr_bandwidth
        # self.l2_capacity = 0 # effective cap here; A100 with dup
        self.l2_bandwidth = 2 *self.ddr_bandwidth
        self.l2_capacity = 6 * (1024**2) # effective cap here; A100 with dup

        self.sm_sub_partitions = 1
        self.l1_smem_throughput_per_cycle = 32*4
        self.configurable_smem_capacity = 64 * (1024**1)
        self.register_capacity_per_sm = 10 * 4
        self.warp_schedulers_per_sm = 1
        # self.fp16_mixed_precision_tflops = 311.87 * 1e12
        # self.fp32_cuda_core_tflops = 19.49 * 1e12
        
        # Now calculate the derived properties
        self.tensor_core_flops = math.prod(self.tensor_core_shape)*2
        self.fp16_mixed_precision_tflops = self.sm_count * self.max_freq * self.tensor_cores_per_sm * self.tensor_core_flops
        self.fp32_cuda_core_tflops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2
        self.smem_bandwidth = self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle
        self.register_bandwidth = self.sm_count * self.max_freq * self.sm_sub_partitions * 64 * 4

