import torch

def reproduce():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping reproduction")
        return

    print(f"CUDA available: {torch.cuda.get_device_name(0)}")

    # Simulate a graph capture
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())

    try:
        print("Starting capture...")
        with torch.cuda.stream(s):
            with torch.cuda.graph(g):
                 t = torch.randn(10, device='cuda')

                 print("Attempting empty_cache inside capture...")
                 torch.cuda.empty_cache()
                 print("empty_cache succeeded (unexpected)")
    except RuntimeError as e:
        print(f"Caught RuntimeError: {e}")
        if "captures_underway" in str(e) or "internal assert" in str(e).lower():
            print("Reproduced the issue!")
    except Exception as e:
        print(f"Caught unexpected exception: {type(e)}: {e}")

if __name__ == "__main__":
    reproduce()
