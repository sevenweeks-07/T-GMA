#include "ThermalAllocator.h"
#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <nvml.h>
#include <vector>

std::atomic<bool> is_running{true};
std::atomic<int> current_temp{0};

// Watchdog Thread: Monitors thermal state and pulls the trigger
void monitor_temperature(ThermalAllocator& allocator) {
    nvmlInit();
    nvmlDevice_t device;
    nvmlDeviceGetHandleByIndex(0, &device);
    
    while (is_running.load()) {
        unsigned int temp;
        nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
        current_temp.store(temp);
        
        std::cout << "[Watchdog] GPU Temp: " << temp << "°C\n";
        
        if (temp >= 34) { // Thermal Threshold
            allocator.defragment();
            std::this_thread::sleep_for(std::chrono::seconds(3));
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    nvmlShutdown();
}

int main() {
    std::cout << "🚀 Booting T-GMA High-Performance Engine...\n";
    ThermalAllocator memory_engine;
    
    // Start the Telemetry Daemon
    std::thread watchdog(monitor_temperature, std::ref(memory_engine));
    
    std::cout << "--- Starting Fragmented Workload Simulation ---\n";
    std::vector<CUdeviceptr> active_tensors;
    int timer = 0;

    // 1. Initial State Check
    memory_engine.log_memory_state(timer++);

    // 2. Stress Test: Allocate 5 Tensors
    for(int i = 0; i < 5; i++) {
        active_tensors.push_back(memory_engine.allocate());
        memory_engine.log_memory_state(timer++);
    }

    // 3. Data Integrity: Write '1337' to the FIRST tensor
    int sentinel = 1337;
    cuMemcpyHtoD(active_tensors[0], &sentinel, sizeof(int));
    std::cout << "[System] Integrity Sentinel '1337' locked into Address " << active_tensors[0] << "\n";

    // 4. Create "Holes" (Fragmentation): Free every other tensor
    std::cout << "[System] Creating artificial fragmentation holes...\n";
    for(size_t i = 1; i < active_tensors.size(); i += 2) {
        memory_engine.free(active_tensors[i]);
        memory_engine.log_memory_state(timer++);
    }

    // 5. Wait for Thermal Trigger and Consolidation
    std::cout << "[System] Waiting for migration to clear fragmentation...\n";
    for(int i = 0; i < 20; i++) { // Increase to 20 seconds
        std::this_thread::sleep_for(std::chrono::seconds(1));
        memory_engine.log_memory_state(timer++);
        
        // Optional: If you see the logs say "Migration Successful", 
        // you know you've captured the data!
    }

    // 6. Verify Integrity after Migration
    int result = 0;
    cuMemcpyDtoH(&result, active_tensors[0], sizeof(int));
    
    std::cout << "\n--- Final System Audit ---\n";
    std::cout << "[Result] Final Data Check: " << result << "\n";
    if (result == sentinel) {
        std::cout << "INTEGRITY PASSED: T-GMA successfully maintained state during migration.\n";
    } else {
        std::cout << "INTEGRITY FAILED: Data corruption in remapped frame.\n";
    }

    std::cout << "[System] Telemetry logged to 'fragmentation_log.csv'.\n";
    std::cout << "Shutting down system...\n";
    
    is_running.store(false);
    watchdog.join();
    
    return 0;
}