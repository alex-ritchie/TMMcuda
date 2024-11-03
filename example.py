# example.py
import torch
import TMMcuda
import time

def compare_matrices(mat1, mat2):
    """Calculate the element-wise difference and percentage accuracy."""
    difference = torch.abs(mat1 - mat2)
    max_diff = difference.max().item()
    accuracy = (difference == 0).sum().item() / difference.numel() * 100
    return max_diff, accuracy

def measure_vram_usage():
    """Measure VRAM usage in bytes."""
    return torch.cuda.memory_allocated()

# Parameters
A_rows, A_cols, B_cols = 64, 128, 64
num_tests = 100

# Variables to accumulate results
baseline_time_total = 0.0
baseline_vram_total = 0
custom_time_total = 0.0
custom_vram_total = 0
accuracy_total = 0.0
max_diff_total = 0.0

for _ in range(num_tests):
    # Generate example matrices
    A = torch.randint(-128, 127, (A_rows, A_cols), dtype=torch.int8, device='cuda')
    B = torch.randint(-1, 2, (A_cols, B_cols), dtype=torch.int8, device='cuda')  # Standard int8 matrix for comparison
    B_compressed = (B + 1).type(torch.uint8)  # Map {-1, 0, 1} to {2, 0, 1} for ternary encoding

    # Standard Matrix Multiplication for Baseline
    torch.cuda.empty_cache()
    start_vram = measure_vram_usage()
    start_time = time.time()

    C_baseline = torch.matmul(A.to(torch.int32), B.to(torch.int32))  # Convert to int32 to avoid overflow
    baseline_time_total += time.time() - start_time
    baseline_vram_total += measure_vram_usage() - start_vram

    # Custom Ternary Matrix Multiplication
    torch.cuda.empty_cache()
    start_vram = measure_vram_usage()
    start_time = time.time()

    C_custom = TMMcuda.tmm(A, B_compressed, A_rows, A_cols, B_cols)
    custom_time_total += time.time() - start_time
    custom_vram_total += measure_vram_usage() - start_vram

    # Accuracy Comparison
    max_diff, accuracy = compare_matrices(C_baseline.to(torch.int8), C_custom)
    max_diff_total += max_diff
    accuracy_total += accuracy

# Calculate averages
baseline_time_avg = baseline_time_total / num_tests
baseline_vram_avg = baseline_vram_total / num_tests
custom_time_avg = custom_time_total / num_tests
custom_vram_avg = custom_vram_total / num_tests
accuracy_avg = accuracy_total / num_tests
max_diff_avg = max_diff_total / num_tests

# Results
print(f"Standard Matrix Multiplication (Average over {num_tests} runs):")
print(f"  Time: {baseline_time_avg:.6f} seconds")
print(f"  VRAM Usage: {baseline_vram_avg / (1024 ** 2):.2f} MB")

print(f"\nCustom Ternary Matrix Multiplication (Average over {num_tests} runs):")
print(f"  Time: {custom_time_avg:.6f} seconds")
print(f"  VRAM Usage: {custom_vram_avg / (1024 ** 2):.2f} MB")

print(f"\nAccuracy of Custom Ternary MatMul (Average over {num_tests} runs):")
print(f"  Average Maximum Element-wise Difference: {max_diff_avg}")
print(f"  Average Percentage of Exact Matches: {accuracy_avg:.2f}%")

