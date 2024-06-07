#include "libs.h"


int main(int argn, char *argv[]){
    
    if(argn < 3){
        std::cerr<<"Not enough parameters.\n";
        std::cerr<<"Proper usage: " << argv[0] << " batch_size cpu/gpu\n";
        return 0;
    }
    int batch_size = atoi(argv[1]);
    char mode = get_mode(argv[2]);
    
    if(mode == 'c' && batch_size > 0){
        printf(BLU "***CPU MODE***\n"    RESET);
        cpu_mode(batch_size);
    }else if(mode == 'g' && batch_size > 0){
        printf(GRN "***GPU MODE***\n"   RESET);
        gpu_mode(batch_size);
    }else{
        std::cerr<<"Unexpected input parameter.\nPlease, provide a proper batch size and the execution mode (gpu/cpu).\n";
    }
    return 0;
}