#include <iostream>
#include <nvml.h>
#include <thread>
#include <atomic>
#include <chrono>

std::atomic<int> current_temp(0); //They allow one thread to write and another to read at the same time without needing a Mutex or causing a crash
void monitor_temp()
{
    nvmlInit();
    nvmlDevice_t device;
    nvmlDeviceGetHandleByIndex(0,&device);

    while(true)
    {
        unsigned int temp;
        if(nvmlDeviceGetTemperature(device,NVML_TEMPERATURE_GPU,&temp)==NVML_SUCCESS)
        {
            current_temp.store(temp);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));//To not overwhelm the cpu

    }
    nvmlShutdown();
}
//store() and load() does action in one single unbreakable step
int main()
{
    std::thread bg_thread(monitor_temp);//start background thread
    bg_thread.detach();//let the thread run on its own

    for (int i = 0; i < 10; i++) {
        std::cout << "Main Thread: The current GPU temp is " << current_temp.load() << " C\n";
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    return 0;

}

