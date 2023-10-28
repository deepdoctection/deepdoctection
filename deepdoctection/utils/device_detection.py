from torch import cuda, backends

def detect_device():

    device_checks = {
        cuda.is_available: "cuda",
        backends.mps.is_available: "mps",
    }

    for check in device_checks:
        if check():
            return device_checks[check]

    else:
        raise ValueError(f'Device not found in {list(device_checks.values())}')