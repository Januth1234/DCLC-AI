# Fix: NVIDIA MX330 - "Windows has stopped this device" (Code 43)

Code 43 usually means a bad or conflicting driver. Try these steps in order.

---

## Step 1: Clean reinstall NVIDIA driver

1. **Uninstall current NVIDIA software**
   - Windows key → Settings → Apps → Installed apps
   - Uninstall anything with "NVIDIA" in the name (GeForce Experience, drivers, etc.)
   - Restart when prompted

2. **Optional – use DDU for a clean removal**
   - Download [Display Driver Uninstaller (DDU)](https://www.guru3d.com/download/display-driver-uninstaller-download/)
   - Boot into **Safe Mode** (hold Shift while clicking Restart → Troubleshoot → Advanced → Startup Settings → Restart → 4)
   - Run DDU → select NVIDIA → "Clean and restart"

3. **Install fresh driver**
   - Go to https://www.nvidia.com/Download/index.aspx
   - Product: GeForce MX330  
   - Get the latest driver (not a beta)
   - Run the installer and choose **Custom install** → **Perform a clean installation**
   - Restart after install

---

## Step 2: If Code 43 is still there

4. **Check in Device Manager**
   - Win+X → Device Manager → Display adapters
   - Right‑click "NVIDIA GeForce MX330"
   - If you see a yellow triangle: Properties → Driver → "Update driver"  
   - Try "Search automatically for drivers"
   - If that fails, "Browse my computer" → "Let me pick" → choose the standard "NVIDIA GeForce MX330" if listed

5. **Try an older driver**
   - Some MX330 systems work better with an older driver
   - NVIDIA driver archive: https://www.nvidia.com/Download/Find.aspx  
   - Try one version a few months older than the latest

6. **Disable then re‑enable**
   - Device Manager → Display adapters
   - Right‑click NVIDIA MX330 → Disable device
   - Restart
   - Device Manager → Right‑click NVIDIA MX330 → Enable device

---

## Step 3: Hardware / BIOS checks

7. **BIOS**
   - Restart → enter BIOS (usually F2, F10, Del, or Esc)
   - Look for "Switchable Graphics", "Discrete GPU", "NVIDIA GPU"
   - Make sure the discrete GPU is enabled (not "Integrated only")
   - Save and exit

8. **Power**
   - Plug in the laptop (power saving can affect the dGPU)
   - Set power plan to "High performance" or "Best performance"

9. **Cables / dock**
   - If using an external monitor or dock, try without them
   - Some docks cause conflicts with hybrid graphics

---

## If it still fails

- **Warranty / hardware**: Code 43 can mean a faulty GPU; consider support or repair.
- **DCLC without GPU**: The project is designed to run on **CPU only** if the GPU stays broken. Training will be slower, but it will work with your i5 and 32 GB RAM.

---

## After the fix

When the GPU is working:

1. Open PowerShell and run: `nvidia-smi` → you should see the MX330
2. In DCLC folder, run: `python scripts/check_gpu.py` → should report CUDA available
3. Set Python to use the GPU: Settings → Display → Graphics → Add `python.exe` → High performance
