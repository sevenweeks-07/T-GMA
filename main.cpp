#include "ThermalAllocator.h"
#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <nvml.h>
#include <cuda.h> // Ensure we have the Driver API for memcpys

std::atomic<bool> is_running{true};
std::atomic<int> current_temp{0};

void monitor_temperature(ThermalAllocator& allocator) {
    nvmlInit();
    nvmlDevice_t device;
    nvmlDeviceGetHandleByIndex(0, &device);
    while (is_running.load()) {
        unsigned int temp;
        nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
        current_temp.store(temp);
        std::cout << "[Watchdog] GPU Temp: " << temp << "°C\n";
        if (temp >= 46) { // Triggering at 46 to ensure we see the migration
            allocator.defragment();
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    nvmlShutdown();
}

int main() {
    std::cout << " Booting Thermal-Aware Allocator Engine...\n";
    ThermalAllocator memory_engine;
    std::thread watchdog(monitor_temperature, std::ref(memory_engine));

    std::cout << "\n--- Simulating ML Workload with Integrity Check ---\n";

    // 1. Allocate space for Tensor A
    CUdeviceptr tensor_a = memory_engine.allocate();

    // 2. WRITE a secret value to Tensor A before migration
    int host_sentinel = 1337;
    cuMemcpyHtoD(tensor_a, &host_sentinel, sizeof(int));
    std::cout << "[Verification] Wrote '" << host_sentinel << "' to Virtual Address " << tensor_a << "\n";

    // 3. Allocate Tensor B to create some weight
    CUdeviceptr tensor_b = memory_engine.allocate();
    
    // 4. Wait for the Watchdog to trigger the 'defragment()' migration
    std::cout << "[Workload] Waiting for thermal trigger and migration...\n";
    std::this_thread::sleep_for(std::chrono::seconds(3));

    // 5. READ the value back from the SAME Virtual Address after migration
    int host_result = 0;
    cuMemcpyDtoH(&host_result, tensor_a, sizeof(int));
    
    std::cout << "\n--- Final Result ---\n";
    std::cout << "[Verification] Read '" << host_result << "' from Virtual Address " << tensor_a << "\n";

    if (host_result == host_sentinel) {
        std::cout << " SUCCESS: Integrity Check Passed! Data survived physical VRAM migration.\n";
    } else {
        std::cout << " FAILURE: Data corruption detected!\n";
    }

    std::cout << "\n Shutting down system...\n";
    is_running.store(false);
    watchdog.join();
    
    return 0;
}