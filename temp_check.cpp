#include <iostream>
#include <nvml.h> 

int main() {
    // 1. Turn on the NVML system. 
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        std::cout << "Failed to initialize NVML: " << nvmlErrorString(result) << "\n";
        return 1;
    }

    // 2. Get a "Handle" to your GPU. 

    nvmlDevice_t device;
    result = nvmlDeviceGetHandleByIndex(0, &device); //Handle is dropped in device and it returns a status code
    if (result != NVML_SUCCESS) {
        std::cout << "Failed to get GPU handle: " << nvmlErrorString(result) << "\n";
        return 1;
    }

    // 3. Ask the GPU for its temperature.
   
    unsigned int temp;
    result = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
    if (result == NVML_SUCCESS) {
        std::cout << "Success! Current GPU Temperature is: " << temp << " C\n";
    } else {
        std::cout << "Failed to read temperature: " << nvmlErrorString(result) << "\n";
    }
    // 4. Turn off the NVML system to clean up.
    nvmlShutdown();

    return 0;
}