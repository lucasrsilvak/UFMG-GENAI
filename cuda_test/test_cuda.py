"""Teste rápido de PyTorch + CUDA na GPU.

Roda uma verificação do ambiente e um benchmark simples de multiplicação
de matrizes comparando CPU vs GPU para comprovar que a GPU está sendo usada.
"""

import time

import torch


def info():
    print("=" * 50)
    print(f"PyTorch version:      {torch.__version__}")
    print(f"CUDA available:     {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("\n CUDA NÃO disponível — PyTorch vai rodar só na CPU.")
        return False
    print(f"CUDA (compilado):    {torch.version.cuda}")
    print(f"GPUs encontradas:    {torch.cuda.device_count()}")
    print(f"GPU 0:               {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"Compute capability:  {cap[0]}.{cap[1]}")
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Memória total:       {total:.1f} GB")
    print("=" * 50)
    return True


def benchmark(n=8000):
    """Multiplica matrizes n x n em CPU e GPU e compara o tempo."""
    print(f"\nBenchmark: multiplicação de matrizes {n}x{n}\n")

    # --- CPU ---
    a_cpu = torch.randn(n, n)
    b_cpu = torch.randn(n, n)
    t0 = time.perf_counter()
    _ = a_cpu @ b_cpu
    t_cpu = time.perf_counter() - t0
    print(f"CPU:  {t_cpu:.3f} s")

    # --- GPU ---
    a_gpu = a_cpu.cuda()
    b_gpu = b_cpu.cuda()
    torch.cuda.synchronize()            # garante que a transferência terminou
    t0 = time.perf_counter()
    _ = a_gpu @ b_gpu
    torch.cuda.synchronize()            # espera o kernel CUDA terminar de verdade
    t_gpu = time.perf_counter() - t0
    print(f"GPU:  {t_gpu:.3f} s")

    if t_gpu > 0:
        print(f"\n GPU foi ~{t_cpu / t_gpu:.1f}x mais rápida que a CPU.")


if __name__ == "__main__":
    if info():
        benchmark()
