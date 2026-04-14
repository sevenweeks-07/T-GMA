#include<cuda.h>

//Single 2MB physical box
struct PageNode{
    CUmemGenericAllocationHandle physical_handle;
    bool is_free;                                 
    PageNode* next;
};

class ThermalAllocator{
private:
    PageNode *head
    CUdeviceptr virtual_base;
    size_t page_size=2*1024*1024;

public:
//Runs when allocator is created
ThermalAllocator(){
    cuMemAddressReserve(&virtual_base, 1024 * 1024 * 1024, page_size, 0, 0); //Reserve 1GB space
}

CUdeviceptr allocate(){
    PageNode* current=head;
    PageNode* previous=nullptr;

    while(current!=nullptr)
    {
        if (current->is_free)
        {
            //Map the physical bed to our virtual room number
            current->is_free=false;
            CUdeviceptr mapped_address=virtual_base+current_offset
            cuMemMap(mapped_address, page_size, 0, current->physical_handle, 0);

            //Set Read/Write permissions so the GPU doesn't crash
            CUmemAccessDesc accessDesc;
            accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            accessDesc.location.id = 0;
            accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            cuMemSetAccess(mapped_address, page_size, &accessDesc, 1);
            
            //Detach the node from the Free List
            if (previous == nullptr) {
                head=current->next;           
            } else {
                previous->next=current->next; 
            }

            current_offset+=page_size

            return mapped_address
        }

        previous=current;
        current=current->next;
    }

    return 0;
}
void free(CUdeviceptr virtual_address) {
        //Find the physical box associated with this address
        PageNode* block = active_allocations[virtual_address];

        //Unmap the memory (sever the link so the room is empty again)
        cuMemUnmap(virtual_address, page_size);

        //Mark the physical box as free
        block->is_free = true;

        //Push it to the VERY FRONT of the Free List
        block->next = head;
        head = block;

        //Remove the record from our dictionary
        active_allocations.erase(virtual_address);
    }
};
