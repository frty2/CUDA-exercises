#include <iostream>
#include <oclUtils.h>
#include <glog/logging.h>

#define width 512
#define height 512

int main(int argc, char ** argv)
{
    cl_int error;
    cl_context context;
    context = clCreateContextFromType(NULL, CL_DEVICE_TYPE_GPU, NULL, NULL, &error);
    
    size_t desc_size;
    clGetContextInfo(context , CL_CONTEXT_DEVICES, NULL, NULL, &desc_size);
    
    cl_device_id* devices = (cl_device_id*) malloc(desc_size);
    clGetContextInfo(context , CL_CONTEXT_DEVICES, desc_size, devices , NULL);
    
    cl_command_queue queue = clCreateCommandQueue(context, devices[0], 0, &error);
    
    
    int kernelsize;
    char* kernelfilepath = shrFindFilePath("mandelbrot.cl", argv[0]);
    const char* kernelsource = oclLoadProgSource(kernelfilepath, "", &kernelsize);
    
    cl_program program = clCreateProgramWithSource(context, 1, &kernelsource, NULL, &error);
    CHECK_EQ(CL_SUCCESS, error) << "Error";
    
    error = clBuildProgram(program, 1, devices, NULL, NULL, NULL);
    
    size_t sizeBuildLog = 200;
    char* buildlog = (char*) malloc(sizeBuildLog);
    size_t copied = 0;
    error = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, sizeBuildLog, buildlog, &copied);
    CHECK_EQ(CL_SUCCESS, error);
    LOG(INFO) << "Build log: " << buildlog;
    
    cl_kernel kernel = clCreateKernel(program, "mandelbrot", &error);
    
    char *result[12];
    int arg = 0;
    error = clSetKernelArg(kernel, arg++, sizeof(char), (void *) &result);
    
    const cl_uint dim = 2;
    size_t globalWorkSize[dim] = {width,height};
    
    error = clEnqueueNDRangeKernel(queue, kernel, dim, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    
    free(buildlog);
    free(devices);
}