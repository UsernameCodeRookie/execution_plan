import numpy as np

def extract_blocks(gridM, gridN, Stride_M, Stride_N):
    # 初始化坐标矩阵，Python中没有直接等价于Matlab中的zeros函数，使用numpy的zeros
    coords = np.zeros((gridM * gridN, 2), dtype=int)
    count = 0
    
    m_start = 1
    n_start = 1
    while m_start <= gridM and n_start <= gridN:
        m_end = min(m_start + Stride_M - 1, gridM)
        n_end = min(n_start + Stride_N - 1, gridN)
        
        for m in range(m_start, m_end + 1):
            for n in range(n_start, n_end + 1):
                coords[count] = [m, n]
                count += 1
                # if (count<=4000):
                #     print(m, n)
                
        
        m_start = m_end + 1
        
        if m_start > gridM:
            m_start = 1
            n_start = n_end + 1
    
    # 调整coords数组的大小为实际使用的大小
    coords = coords[:count]
    
    return coords

# # 示例用法
# gridM = 5  # 示例参数，应根据实际情况进行调整
# gridN = 5  # 示例参数，应根据实际情况进行调整
# Stride_M = 2  # 示例参数，应根据实际情况进行调整
# Stride_N = 2  # 示例参数，应根据实际情况进行调整
# coords = extract_blocks(gridM, gridN, Stride_M, Stride_N)
# print(coords)


