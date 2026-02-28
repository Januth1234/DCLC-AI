"""Attempts to fix or improve discrete GPU (MX330) visibility for PyTorch."""
import os
import sys
import subprocess

def set_gpu_preferred():
    """Set CUDA to prefer NVIDIA GPU when both Intel and NVIDIA exist."""
    # On machines with hybrid graphics, this can help
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU (NVIDIA when visible)
    # Some systems need this to avoid Intel OpenCL taking over
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

def check_and_fix():
    print("DCLC GPU Fix - Attempting to enable MX330 for PyTorch\n")
    
    set_gpu_preferred()
    
    # Re-import torch after env vars (in case torch was imported before)
    import importlib
    if "torch" in sys.modules:
        importlib.reload(__import__("torch"))
    
    import torch
    
    if torch.cuda.is_available():
        print("[OK] CUDA is available!")
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"     GPU {i}: {p.name} ({p.total_memory/1024**2:.0f} MB)")
        return True
    
    print("[NOT FIXED] CUDA still not available.")
    print("\nManual steps (you must do these):")
    print("1. Right-click Desktop > Display settings > Graphics")
    print("   (or: Windows Settings > System > Display > Graphics)")
    print("2. Click 'Add an app' > Browse")
    print("3. Add: python.exe (from your Python install, e.g. C:\\Users\\...\\python.exe)")
    print("4. Set it to 'High performance' (NVIDIA)")
    print("5. Restart your terminal/IDE and run this script again")
    print("\nAlternative: Run the whole terminal as Administrator, then try again.")
    return False

if __name__ == "__main__":
    success = check_and_fix()
    sys.exit(0 if success else 1)
