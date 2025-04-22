import torch

def main():

    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available. You can use GPU for training.")
    else:
        print("CUDA is not available. You will be using CPU for training.")

    # Check the number of GPUs available
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # Print the name of the GPU being used
    if num_gpus > 0:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {gpu_name}")
    else:
        print("No GPUs found.")

    # Check the current device
    current_device = torch.cuda.current_device() if num_gpus > 0 else "CPU"
    print(f"Current device: {current_device}")

    # Print CUDA and cuDNN versions
    print(f"torch.version.cuda : {torch.version.cuda}")
    print(f"torch.backends.cudnn.version() : {torch.backends.cudnn.version()}")
