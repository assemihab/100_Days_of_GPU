# %%
import pycuda
import pycuda.driver as drv

# %%
drv.init()

# %%

drv.Device(0)

# %%
for i in range (drv.Device.count()):
    gpu_device = drv.Device(i)
    compute_capability = float('%d.%d' % gpu_device.compute_capability())
    device_attributes_tuples = gpu_device.get_attributes().items()
    device_attributes = {}
    for k, v in device_attributes_tuples:
        device_attributes[str(k)] = v
    num_mp = device_attributes['MULTIPROCESSOR_COUNT']
    print("Device #%d: %s" % (i, gpu_device.name()))
    cuda_cores_per_mp = {5.0: 128, 5.1: 128, 5.2: 128, 6.0: 64, 6.1: 128, 6.2: 128, 7.2: 64, 7.5: 64,8.9:128}[compute_capability]
    print("Number of CUDA Cores: ", cuda_cores_per_mp * num_mp)
    print("\t Compute Capability: %d.%d" % gpu_device.compute_capability())
    print("\t Total Memory: %s mb" % (gpu_device.total_memory()//(1024**2)))
    # print("Number of Streaming Multiprocessors: ", num_mp)
    for k in sorted(device_attributes.keys()):
        print("\t %s: %s" % (k, device_attributes[k]))
    print("\n")
    

# %%



