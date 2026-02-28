"""Diagnostic script for NVIDIA GPU detection. Helps troubleshoot MX330 not showing in Task Manager."""
import sys
import subprocess
import platform

def main():
    print("=" * 60)
    print("DCLC GPU Diagnostic - Checking discrete GPU (MX330)")
    print("=" * 60)
    
    # 1. PyTorch CUDA check
    print("\n[1] PyTorch CUDA status:")
    try:
        import torch
        print(f"    PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"    CUDA available: YES")
            print(f"    CUDA version: {torch.version.cuda}")
            print(f"    GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"    GPU {i}: {props.name} ({props.total_memory / 1024**2:.0f} MB)")
        else:
            print(f"    CUDA available: NO")
    except ImportError:
        print("    PyTorch not installed - run: pip install torch")
    
    # 2. nvidia-smi (system-level check)
    print("\n[2] nvidia-smi (system driver check):")
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"    nvidia-smi failed (code {result.returncode})")
            if result.stderr:
                print(f"    Stderr: {result.stderr[:200]}")
    except FileNotFoundError:
        print("    nvidia-smi NOT FOUND - NVIDIA drivers may not be installed or not in PATH")
    except Exception as e:
        print(f"    Error: {e}")
    
    # 3. Windows Device Manager check (via wmic/powershell)
    if platform.system() == "Windows":
        print("\n[3] Windows display adapters (PowerShell):")
        try:
            result = subprocess.run(
                ["powershell", "-Command", 
                 "Get-WmiObject Win32_VideoController | Select-Object Name, AdapterRAM, DriverVersion | Format-List"],
                capture_output=True,
                text=True,
                timeout=15,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            if result.returncode == 0 and result.stdout.strip():
                print(result.stdout)
            else:
                print("    Could not query display adapters")
        except Exception as e:
            print(f"    Error: {e}")
    
    # 4. Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS:")
    print("=" * 60)
    
    cuda_ok = False
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
    except ImportError:
        pass
    
    if cuda_ok:
        print("  [OK] GPU is detected and usable by PyTorch.")
    else:
        print("  [ACTION NEEDED] GPU not detected. Try:")
        print("    1. Install/update NVIDIA drivers: https://www.nvidia.com/Download/index.aspx")
        print("    2. Windows: Settings > Display > Graphics > Add your Python executable")
        print("       Set it to 'High performance' (use discrete GPU)")
        print("    3. Laptop: Check power plan - some throttle GPU on battery")
        print("    4. BIOS: Look for 'Discrete GPU', 'Switchable Graphics', or 'NVIDIA GPU'")
        print("       Ensure it's enabled (not 'Integrated only')")
        print("    5. Reboot after driver install")

if __name__ == "__main__":
    main()
