import torch

def get_best_gpu_device(min_free_memory_gb: float = 2.0) -> str:
    """Automatically select the best available GPU device."""
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return "cpu"
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("No GPUs found, using CPU")
        return "cpu"
    
    print(f"Found {num_gpus} GPU(s), checking availability...")
    
    best_device = None
    best_free_memory = 0
    
    for i in range(num_gpus):
        try:
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / (1024**3)
            
            torch.cuda.set_device(i)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            free_memory = total_memory - reserved
            
            print(f"  GPU {i} ({props.name}): {free_memory:.1f}GB free / {total_memory:.1f}GB total")
            
            if free_memory >= min_free_memory_gb and free_memory > best_free_memory:
                best_device = f"cuda:{i}"
                best_free_memory = free_memory
                
        except Exception as e:
            print(f"  GPU {i}: Error checking - {e}")
            continue
    
    if best_device is not None:
        print(f"Selected device: {best_device} ({best_free_memory:.1f}GB free)")
        return best_device
    else:
        print(f"No GPU with >= {min_free_memory_gb}GB free memory found, using CPU")
        return "cpu"
