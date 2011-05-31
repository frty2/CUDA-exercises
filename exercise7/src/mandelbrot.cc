#if __APPLE__
    #include <OpenCL/cl.h>
#else
     #include <CL/cl.h>
#endif
#include <iterator>
#include <string>
#include <iostream>
#include <fstream>
#include <glog/logging.h>
#include <stdlib.h>

#include "types.h"
#include "mainwindow.h"
#include "ppm_writer.h"

#define width 512
#define height 512

std::string readFile(const char* filename)
{
    std::ifstream in(filename);
    return std::string((std::istreambuf_iterator<char>(in)),
                       std::istreambuf_iterator<char>());
}

void save(char *result, int argc, char ** argv)
{
    rgb color;
    rgb *image = (rgb*) malloc(width * height * sizeof(rgb));
    CHECK_NOTNULL(image);

    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            if(result[y * width + x] == 100)
            {
                color.x = 0;
                color.y = 0;
                color.z = 0;
            }
            else
            {
                color.x = 255 - 2 * result[y * width + x];
                color.y = 255 - 2 * result[y * width + x];
                color.z = 255 - 2 * result[y * width + x];
            }
            image[y * width + x] = color;
        }
    }
    const char* filename = "mandelbrot.ppm";
    write_ppm(image, width, height, filename);
    //displayimage(argc, argv, image, width, height);
    free(image);
}

/**
 * found at http://forums.amd.com/forum/messageview.cfm?catid=390&threadid=128536
 */
char *print_cl_errstring(cl_int err)
{
    switch (err)
    {
    case CL_SUCCESS:
        return strdup("Success!");
    case CL_DEVICE_NOT_FOUND:
        return strdup("Device not found.");
    case CL_DEVICE_NOT_AVAILABLE:
        return strdup("Device not available");
    case CL_COMPILER_NOT_AVAILABLE:
        return strdup("Compiler not available");
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        return strdup("Memory object allocation failure");
    case CL_OUT_OF_RESOURCES:
        return strdup("Out of resources");
    case CL_OUT_OF_HOST_MEMORY:
        return strdup("Out of host memory");
    case CL_PROFILING_INFO_NOT_AVAILABLE:
        return strdup("Profiling information not available");
    case CL_MEM_COPY_OVERLAP:
        return strdup("Memory copy overlap");
    case CL_IMAGE_FORMAT_MISMATCH:
        return strdup("Image format mismatch");
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
        return strdup("Image format not supported");
    case CL_BUILD_PROGRAM_FAILURE:
        return strdup("Program build failure");
    case CL_MAP_FAILURE:
        return strdup("Map failure");
    case CL_INVALID_VALUE:
        return strdup("Invalid value");
    case CL_INVALID_DEVICE_TYPE:
        return strdup("Invalid device type");
    case CL_INVALID_PLATFORM:
        return strdup("Invalid platform");
    case CL_INVALID_DEVICE:
        return strdup("Invalid device");
    case CL_INVALID_CONTEXT:
        return strdup("Invalid context");
    case CL_INVALID_QUEUE_PROPERTIES:
        return strdup("Invalid queue properties");
    case CL_INVALID_COMMAND_QUEUE:
        return strdup("Invalid command queue");
    case CL_INVALID_HOST_PTR:
        return strdup("Invalid host pointer");
    case CL_INVALID_MEM_OBJECT:
        return strdup("Invalid memory object");
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
        return strdup("Invalid image format descriptor");
    case CL_INVALID_IMAGE_SIZE:
        return strdup("Invalid image size");
    case CL_INVALID_SAMPLER:
        return strdup("Invalid sampler");
    case CL_INVALID_BINARY:
        return strdup("Invalid binary");
    case CL_INVALID_BUILD_OPTIONS:
        return strdup("Invalid build options");
    case CL_INVALID_PROGRAM:
        return strdup("Invalid program");
    case CL_INVALID_PROGRAM_EXECUTABLE:
        return strdup("Invalid program executable");
    case CL_INVALID_KERNEL_NAME:
        return strdup("Invalid kernel name");
    case CL_INVALID_KERNEL_DEFINITION:
        return strdup("Invalid kernel definition");
    case CL_INVALID_KERNEL:
        return strdup("Invalid kernel");
    case CL_INVALID_ARG_INDEX:
        return strdup("Invalid argument index");
    case CL_INVALID_ARG_VALUE:
        return strdup("Invalid argument value");
    case CL_INVALID_ARG_SIZE:
        return strdup("Invalid argument size");
    case CL_INVALID_KERNEL_ARGS:
        return strdup("Invalid kernel arguments");
    case CL_INVALID_WORK_DIMENSION:
        return strdup("Invalid work dimension");
    case CL_INVALID_WORK_GROUP_SIZE:
        return strdup("Invalid work group size");
    case CL_INVALID_WORK_ITEM_SIZE:
        return strdup("Invalid work item size");
    case CL_INVALID_GLOBAL_OFFSET:
        return strdup("Invalid global offset");
    case CL_INVALID_EVENT_WAIT_LIST:
        return strdup("Invalid event wait list");
    case CL_INVALID_EVENT:
        return strdup("Invalid event");
    case CL_INVALID_OPERATION:
        return strdup("Invalid operation");
    case CL_INVALID_GL_OBJECT:
        return strdup("Invalid OpenGL object");
    case CL_INVALID_BUFFER_SIZE:
        return strdup("Invalid buffer size");
    case CL_INVALID_MIP_LEVEL:
        return strdup("Invalid mip-map level");
    default:
        return strdup("Unknown");
    }
}

//include <cstring>, etc.
cl_int getPlatformID(cl_platform_id* clSelectedPlatformID) {
    char chBuffer[1024];
    cl_uint num_platforms;
    cl_platform_id* clPlatformIDs;
    cl_int ciErrNum;
    *clSelectedPlatformID = NULL;

    // Get OpenCL platform count
    ciErrNum = clGetPlatformIDs (0, NULL, &num_platforms);
    if (ciErrNum != CL_SUCCESS) {
        return -1000;
    } else {
        if(num_platforms == 0) {
            return -2000;
        } else {
            // if there's a platform or more, make space for ID's
            if ((clPlatformIDs = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id))) == NULL) {
                return -3000;
            }

            // get platform info for each platform and trap the NVIDIA platform if found
            ciErrNum = clGetPlatformIDs (num_platforms, clPlatformIDs, NULL);
            for(cl_uint i = 0; i < num_platforms; ++i) {
                ciErrNum = clGetPlatformInfo (clPlatformIDs[i], CL_PLATFORM_NAME, 1024, &chBuffer, NULL);
                if(ciErrNum == CL_SUCCESS) {
                    if(strstr(chBuffer, "NVIDIA") != NULL) {
                        *clSelectedPlatformID = clPlatformIDs[i];
                        break;
                    }
                }
            }

            // default to zeroeth platform if NVIDIA not found
            if(*clSelectedPlatformID == NULL) {
                *clSelectedPlatformID = clPlatformIDs[0];
            }

            free(clPlatformIDs);
        }
    }
    return CL_SUCCESS;
}

int main(int argc, char ** argv)
{
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();

    cl_int error;
    cl_context context;
    cl_device_id *devices;
    
    #if __APPLE__
        context = clCreateContextFromType(NULL, CL_DEVICE_TYPE_GPU, NULL, NULL, &error);
        CHECK_EQ(error, CL_SUCCESS) << print_cl_errstring(error) << std::endl;
    #else
    cl_uint num_platforms;
    cl_platform_id* clPlatformIDs;
    error = clGetPlatformIDs (0, NULL, &num_platforms);
    CHECK_EQ(error, CL_SUCCESS) << print_cl_errstring(error) << std::endl;
    if(num_platforms == 0)
    {
        LOG(ERROR) << "No platforms found" << std::endl;
        return -1;
    }
    clPlatformIDs = (cl_platform_id*) malloc(num_platforms*sizeof(cl_platform_id));
    CHECK_NOTNULL(clPlatformIDs);
    
    error = clGetPlatformIDs (num_platforms, clPlatformIDs, NULL);
    char chBuffer[1024];
    cl_platform_id platform = NULL;
    bool found = false;
    
    for(cl_uint i = 0; i < num_platforms; ++i) {
        error = clGetPlatformInfo (clPlatformIDs[i], CL_PLATFORM_NAME, 1024, &chBuffer, NULL);
        if(error == CL_SUCCESS) {
            if(strstr(chBuffer, "NVIDIA") != NULL) {
                platform = clPlatformIDs[i];
                found = true;
                break;
            }
        }
    }
    if(! found){
        LOG(ERROR) << "No nvidia plattform found" << std::endl;
        return -1;
    }
    
    cl_uint num_devices;
    
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
    CHECK_EQ(error, CL_SUCCESS) << print_cl_errstring(error) << std::endl;
    if(num_devices == 0)
    {
        LOG(ERROR) << "No devices found" << std::endl;
        return -1;
    }
    devices = (cl_device_id*) malloc(num_devices*sizeof(cl_device_id));
    CHECK_NOTNULL(devices);
    
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);
    CHECK_EQ(error, CL_SUCCESS) << print_cl_errstring(error) << std::endl;
    
    cl_context_properties props[3];
    props[0] = (cl_context_properties)CL_CONTEXT_PLATFORM;
    props[1] = (cl_context_properties)platform;
    props[2] = (cl_context_properties)0;
    
    
    context = clCreateContext(props, num_devices, devices, NULL, NULL, &error);
    CHECK_EQ(error, CL_SUCCESS) << print_cl_errstring(error) << std::endl;
    
    #endif

    size_t desc_size;
    clGetContextInfo(context , CL_CONTEXT_DEVICES, NULL, NULL, &desc_size);

    devices = (cl_device_id*) malloc(desc_size);
    clGetContextInfo(context , CL_CONTEXT_DEVICES, desc_size, devices , NULL);

    cl_command_queue queue = clCreateCommandQueue(context, devices[0], 0, &error);
    CHECK_EQ(error, CL_SUCCESS) << print_cl_errstring(error) << std::endl;

    const char* kernelsource = readFile("../src/mandelbrot.cl").c_str();

    cl_program program = clCreateProgramWithSource(context, 1, &kernelsource, NULL, &error);
    CHECK_EQ(error, CL_SUCCESS) << print_cl_errstring(error) << std::endl;

    error = clBuildProgram(program, 1, devices, NULL, NULL, NULL);

    size_t sizeBuildLog = 2000;
    char* buildlog = (char*) malloc(sizeBuildLog);
    size_t copied = 0;
    error = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, sizeBuildLog, buildlog, &copied);
    CHECK_EQ(error, CL_SUCCESS) << print_cl_errstring(error) << std::endl;

    std::cout << "Build log: " << buildlog << std::endl;

    cl_kernel kernel = clCreateKernel(program, "mandelbrot", &error);
    CHECK_EQ(error, CL_SUCCESS) << print_cl_errstring(error) << std::endl;

    cl_mem device_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(char) * width * height, NULL, NULL);

    int arg = 0;
    error = clSetKernelArg(kernel, arg++, sizeof(device_result), (void *) &device_result);
    CHECK_EQ(error, CL_SUCCESS) << print_cl_errstring(error) << std::endl;

    const cl_uint dim = 2;
    size_t globalWorkSize[dim] = {width, height};

    error = clEnqueueNDRangeKernel(queue, kernel, dim, NULL, globalWorkSize, NULL, 0, NULL, NULL);
    CHECK_EQ(error, CL_SUCCESS) << print_cl_errstring(error) << std::endl;

    char *result = (char *) malloc(width * height * sizeof(char));
    CHECK_NOTNULL(result);
    clEnqueueReadBuffer(queue, device_result, CL_TRUE, 0, width * height * sizeof(char), result, 0, NULL, NULL);

    save(result, argc, argv);
    
    clReleaseMemObject(device_result);
    free(buildlog);
    free(devices);
    free(result);
    #if __APPLE__
    #else
        free(devices);
        free(clPlatformIDs);
    #endif
}
