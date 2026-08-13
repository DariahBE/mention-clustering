"""
            CONVENIENCE UTILITY THAT USES THE NVIDIA-SMI TOOL TO VERIFY THE
            AVAILABILITY OF A GPU, LOOK WHICH GPU ARE NOT USED AND AUTOMATICALLY
            LIMIT THE SYSTEM TO USE THE NON-USED GPU USING ENVIRON SETTINGS.
"""

import os
import subprocess
from typing import List, Optional

def pick_gpu(verbosity: int = 1, mode: str = "auto", max_gpus: int = 1) -> Optional[List[int]]:
    """
    Args:
        verbosity (int): Level of logging. 0 = silent, 1 = info, 2 = debug.
        mode (str): 'report' to just return available GPUs,
                    'auto' to automatically set CUDA_VISIBLE_DEVICES.
        max_gpus: limits the amount of registered gpus to the given amount. 

    Returns:
        List[int] if mode == "report", else None.
    """
    try:
        # Get list of all GPU indices and UUIDs
        all_gpus_cmd = ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"]
        all_gpus_output = subprocess.check_output(all_gpus_cmd, encoding="utf-8").strip().splitlines()
        gpu_map = {}
        for line in all_gpus_output:
            idx, uuid = line.strip().split(", ")
            gpu_map[uuid] = int(idx)

        all_gpu_indices = sorted(gpu_map.values())

        # Get GPUs currently in use by checking running processes
        used_gpus_cmd = ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"]
        used_gpus_output = subprocess.check_output(used_gpus_cmd, encoding="utf-8").strip()
        used_gpu_uuids = [uuid.strip() for uuid in used_gpus_output.splitlines() if uuid.strip()]

        # Free GPUs = all - used
        used_gpu_indices = {gpu_map[uuid] for uuid in used_gpu_uuids if uuid in gpu_map}
        free_gpus = [idx for idx in all_gpu_indices if idx not in used_gpu_indices]


        if verbosity > 0:
            print(f"[INFO] Found {len(free_gpus)} free GPUs: {free_gpus}")

        if mode == "report":
            return free_gpus

        elif mode == "auto":
            if free_gpus:
                # Limit number of GPUs if requested
                if max_gpus > 0:
                    free_gpus = free_gpus[:max_gpus]
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, free_gpus))
                if verbosity > 0:
                    print(f"[INFO] Using GPUs: {free_gpus}")
            else:
                if verbosity > 0:
                    print("[WARN] No free GPUs found. CUDA_VISIBLE_DEVICES not set.")

        else:
            raise ValueError("mode must be either 'report' or 'auto'")

    except FileNotFoundError:
        if verbosity > 0:
            print("[ERROR] nvidia-smi not found. Ensure NVIDIA drivers are installed.")
        return [] if mode == "report" else None

    except subprocess.CalledProcessError as e:
        if verbosity > 0:
            print(f"[ERROR] Failed to run nvidia-smi: {e}")
        return [] if mode == "report" else None


if __name__ == "__main__": 
    print("testing")
    pick_gpu()
