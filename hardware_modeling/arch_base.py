# 文件名: arch_base.py
class Arch:
    def __init__(self):
        self.core = None
        self.sm_count = 0
        self.base_freq = 0.0
        self.max_freq = 0.0
        self.tensor_cores_per_sm = 0
        self.tensor_core_shape = (0, 0, 0)
        self.tensor_core_flops = 0
        self.fp32_cores_per_sm = 0
        self.ddr_bandwidth = 0.0
        self.ddr_capacity = 0
        self.l2_bandwidth = 0.0
        self.l2_capacity = 0
        self.sm_sub_partitions = 0
        self.l1_smem_throughput_per_cycle = 0
        self.configurable_smem_capacity = 0
        self.register_capacity_per_sm = 0
        self.warp_schedulers_per_sm = 0
        self.fp16_mixed_precision_tflops = 0.0
        self.fp32_cuda_core_tflops = 0.0
        self.smem_bandwidth = 0
        self.register_bandwidth = 0
        # self.smem_bandwidth = self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle
        # self.register_bandwidth = self.sm_count * self.max_freq * self.sm_sub_partitions * 32 * 4

        self.ddr_max_util=0.85
        self.l2_max_util=0.85
        self.l1_max_util=0.9
        self.compute_max_util=0.9



    # ... other methods as needed
