import torch
import sys

print("="*60)
print("CUDA 环境诊断")
print("="*60)

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version (PyTorch built with): {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU count: {torch.cuda.device_count()}")
    
    # 测试基本 CUDA 操作
    print("\n" + "="*60)
    print("测试基本 CUDA 操作")
    print("="*60)
    
    # 测试 1: 简单张量创建
    print("\n[TEST 1] 创建 CUDA 张量...")
    try:
        a = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        print(f"  成功: {a}")
    except Exception as e:
        print(f"  失败: {e}")
    
    # 测试 2: 张量加法
    print("\n[TEST 2] 张量加法...")
    try:
        a = torch.tensor([1.0, 2.0], device='cuda')
        b = torch.tensor([3.0, 4.0], device='cuda')
        c = a + b
        torch.cuda.synchronize()
        print(f"  成功: {c}")
    except Exception as e:
        print(f"  失败: {e}")
    
    # 测试 3: 简单矩阵乘法 (不用 cuBLAS)
    print("\n[TEST 3] 元素级乘法...")
    try:
        a = torch.tensor([1.0, 2.0], device='cuda')
        b = torch.tensor([3.0, 4.0], device='cuda')
        c = a * b
        torch.cuda.synchronize()
        print(f"  成功: {c}")
    except Exception as e:
        print(f"  失败: {e}")
    
    # 测试 4: matmul (需要 cuBLAS) - 向量点积
    print("\n[TEST 4] 向量点积 (torch.dot)...")
    try:
        a = torch.tensor([1.0, 2.0], device='cuda')
        b = torch.tensor([3.0, 4.0], device='cuda')
        c = torch.dot(a, b)
        torch.cuda.synchronize()
        print(f"  成功: {c}")
    except Exception as e:
        print(f"  失败: {e}")
    
    # 测试 5: matmul - 矩阵向量乘法 (使用 cuBLAS)
    print("\n[TEST 5] 矩阵向量乘法 (torch.mv)...")
    try:
        A = torch.eye(2, device='cuda', dtype=torch.float32)
        x = torch.ones(2, device='cuda', dtype=torch.float32)
        y = torch.mv(A, x)
        torch.cuda.synchronize()
        print(f"  成功: {y}")
    except Exception as e:
        print(f"  失败: {e}")
    
    # 测试 5b: 使用 matmul 替代 mv
    print("\n[TEST 5b] 矩阵向量乘法 (用 matmul 替代 mv)...")
    try:
        A = torch.eye(2, device='cuda', dtype=torch.float32)
        x = torch.ones(2, 1, device='cuda', dtype=torch.float32)  # 变成列向量
        y = torch.matmul(A, x).squeeze()
        torch.cuda.synchronize()
        print(f"  成功: {y}")
    except Exception as e:
        print(f"  失败: {e}")
    
    # 测试 6: matmul - 矩阵乘法 (使用 cuBLAS)
    print("\n[TEST 6] 矩阵乘法 (torch.matmul)...")
    try:
        A = torch.eye(2, device='cuda', dtype=torch.float32)
        B = torch.ones(2, 2, device='cuda', dtype=torch.float32)
        C = torch.matmul(A, B)
        torch.cuda.synchronize()
        print(f"  成功:\n{C}")
    except Exception as e:
        print(f"  失败: {e}")
    
    # 测试 7: 使用 @ 运算符
    print("\n[TEST 7] @ 运算符 (4x4 @ 4 向量)...")
    try:
        A = torch.eye(4, device='cuda', dtype=torch.float32)
        x = torch.tensor([0., 0., 0., 1.], device='cuda', dtype=torch.float32)
        y = A @ x
        torch.cuda.synchronize()
        print(f"  成功: {y}")
    except Exception as e:
        print(f"  失败: {e}")
    
    # 测试 7b: @ 运算符，但把向量变成 2D
    print("\n[TEST 7b] @ 运算符 (4x4 @ 4x1 矩阵)...")
    try:
        A = torch.eye(4, device='cuda', dtype=torch.float32)
        x = torch.tensor([[0.], [0.], [0.], [1.]], device='cuda', dtype=torch.float32)  # 4x1
        y = (A @ x).squeeze()
        torch.cuda.synchronize()
        print(f"  成功: {y}")
    except Exception as e:
        print(f"  失败: {e}")
    
    # 测试 7c: 使用 torch.mm 替代
    print("\n[TEST 7c] torch.mm (4x4 @ 4x1)...")
    try:
        A = torch.eye(4, device='cuda', dtype=torch.float32)
        x = torch.tensor([[0.], [0.], [0.], [1.]], device='cuda', dtype=torch.float32)
        y = torch.mm(A, x).squeeze()
        torch.cuda.synchronize()
        print(f"  成功: {y}")
    except Exception as e:
        print(f"  失败: {e}")
    
    # 测试 7d: 使用 einsum 替代
    print("\n[TEST 7d] torch.einsum 替代矩阵向量乘法...")
    try:
        A = torch.eye(4, device='cuda', dtype=torch.float32)
        x = torch.tensor([0., 0., 0., 1.], device='cuda', dtype=torch.float32)
        y = torch.einsum('ij,j->i', A, x)
        torch.cuda.synchronize()
        print(f"  成功: {y}")
    except Exception as e:
        print(f"  失败: {e}")
    
    # 测试 8: CPU 上的 matmul（对比测试）
    print("\n[TEST 8] CPU 上的 matmul...")
    try:
        A = torch.eye(4, dtype=torch.float32)
        x = torch.tensor([0., 0., 0., 1.], dtype=torch.float32)
        y = torch.matmul(A, x)
        print(f"  成功: {y}")
    except Exception as e:
        print(f"  失败: {e}")

else:
    print("CUDA 不可用！")

print("\n" + "="*60)
print("诊断完成")
print("="*60)