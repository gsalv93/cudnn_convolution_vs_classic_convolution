main: main.cu functions.cu gpu_functions.cu cpu_functions.cu libs.h
	nvcc main.cu -o main functions.cu gpu_functions.cu cpu_functions.cu -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lcudnn -I.