import os
import argparse
import torch
import numpy as np
from thop import profile
from thop import clever_format

# Import the core model
from net.MyDiagX import MyDiag21 as create_model

def measure_latency_and_throughput(model, device, input_shape=(1, 3, 224, 224), repetitions=300):
    """
    Measure actual inference latency and throughput using the strict CUDA Event mechanism.
    """
    dummy_input = torch.randn(*input_shape, dtype=torch.float).to(device)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((repetitions, 1))
    
    print(f"Measuring latency over {repetitions} iterations...")
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # Synchronize the GPU and wait for the current operation to complete
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
            
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    
    # Throughput calculation: 1000 ms / average latency (ms) = number of images processed per second (batch_size=1)
    throughput = 1000 / mean_syn
    
    return mean_syn, std_syn, throughput

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("WARNING: You are running on CPU. For accurate hardware efficiency (Latency/Throughput), a GPU is strongly recommended.")
    
    print(f"Hardware Evaluation on Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    
    # Initialize the model
    num_classes = 4 # For the sake of simplicity, we will use the BTD-4 as an example.
    model = create_model(num_classes=num_classes)
    model.to(device)
    model.eval()

    # Standard input size (224x224)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    # ==========================================
    # 1. Number of Computational Parameters and Computational Complexity (FLOPs)
    # ==========================================
    print("\nCalculating Params and FLOPs...")

    macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
    macs_formatted, params_formatted = clever_format([macs, params], "%.3f")

    # ==========================================
    # 2. Measuring Latency and Throughput
    # ==========================================
    latency_mean, latency_std, throughput = measure_latency_and_throughput(model, device)

    # ==========================================
    # 3. Measuring Memory Usage (Memory Footprint)
    # ==========================================
    if device.type == 'cuda':
        memory_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2) # 转换为 MB
    else:
        memory_allocated = 0.0

    # ==========================================
    # Print the final summary report
    # ==========================================
    print("\n" + "="*50)
    print("HARDWARE EFFICIENCY REPORT")
    print("="*50)
    print(f"Model Name           : BTNet-TS")
    print(f"Input Resolution     : 3 x 224 x 224")
    print(f"Parameters (Params)  : {params_formatted} ({params / 1e6:.2f} M)")
    print(f"Complexity (MACs)    : {macs_formatted} ({macs / 1e9:.2f} G)")
    print(f"Inference Latency    : {latency_mean:.2f} ± {latency_std:.2f} ms / image")
    print(f"Throughput           : {throughput:.2f} images / second")
    if device.type == 'cuda':
        print(f"Peak GPU Memory      : {memory_allocated:.2f} MB")
    print("="*50)
    print("Note: FLOPs is often reported identically to MACs in PyTorch literature.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate hardware efficiency of BTNet-TS.")
    args = parser.parse_args()
    main(args)
