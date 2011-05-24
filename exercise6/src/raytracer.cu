#include <cmath>
#include <glog/logging.h>
#include <iostream>
#include <sys/time.h>

#include "float.h"
#include "raytracer.h"

#define EPSILON1 0.0001f
#define EPSILON2 0.00000001f
#define PI 3.14159265358f

#define chunksizex 20
#define chunksizey 16
struct rtconfig
{
    rgb background;
    camera cam;
    int width;
    int height;
    point xgap;
    point ygap;
    point upperleft;
    int lightcount;
    int tricount;
};
typedef struct rtconfig rtconfig;


__host__ __device__ point cross(const point& p1, const point& p2)
{
    point result;
    result.x = p1.y * p2.z - p1.z * p2.y;
    result.y = p1.z * p2.x - p1.x * p2.z;
    result.z = p1.x * p2.y - p1.y * p2.x;
    return result;
}

__host__ __device__ float dot(const point& p1, const point& p2)
{
    return p1.x * p2.x + p1.y * p2.y + p1.z * p2.z;
}

__host__ __device__ float norm(const point& p)
{
    return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

__host__ __device__ float anglebetween(const point& p1, const point& p2)
{
    return dot(p1, p2) / (norm(p1) * norm(p2));
}

__host__ __device__ void normalize(point& p)
{
    float len = norm(p);
    if(len != 0.0f)
    {
        p.x /= len;
        p.y /= len;
        p.z /= len;
    }
}

__host__ __device__ point operator*(const point& vector, float scalar)
{
    point result;
    result.x = scalar * vector.x;
    result.y = scalar * vector.y;
    result.z = scalar * vector.z;
    return result;
}

__host__ __device__ point operator*(float scalar, const point& vector)
{
    return vector * scalar;
}

__host__ __device__ point operator+(const point& left, const point& right)
{
    point result;
    result.x = left.x + right.x;
    result.y = left.y + right.y;
    result.z = left.z + right.z;

    return result;
}

__host__ __device__ point operator-(const point& left, const point& right)
{
    point result;
    result.x = left.x - right.x;
    result.y = left.y - right.y;
    result.z = left.z - right.z;

    return result;
}

__host__ __device__ rgb shade(const rgb& color, float fraction)
{
    if(fraction > 1.0f) { fraction = 1.0f; }
    rgb result;
    result.x = color.x * fraction;
    result.z = color.z * fraction;
    result.y = color.y * fraction;
    return result;
}

__host__ __device__ bool intersect(const point& location, const point& direction, const point& normal, const point& p, point& intersection)
{
    float t = dot(normal, p - location) / dot(normal, direction);

    intersection = location + t * direction;
    return t >= EPSILON1;
}

// checks if point p is on the same side of the line AB as C
__host__ __device__ bool inside(const point& p, const point& c, const point& a, const point& b)
{
    return dot(cross(b - a, p - a), cross(b - a, c - a)) >= -EPSILON2;
}

__host__ __device__ bool intersect(const ray& r, const triangle& t, point& intersection)
{
    //calc intersection with triangle surface
    point normal = cross(t.A - t.B, t.A - t.C);
    normalize(normal);
    return intersect(r.location, r.direction, normal, t.A, intersection) &&
           inside(intersection, t.A, t.B, t.C) &&
           inside(intersection, t.B, t.A, t.C) &&
           inside(intersection, t.C, t.A, t.B);
}
//woop testen
__host__ __device__ bool fastintersect(const ray& r, const triangle& tri, point& intersection)
{
    float t, u, v;
    float det, inv_det;
    point tvec, pvec, qvec;

    point edge1 = tri.B - tri.A;
    point edge2 = tri.C - tri.A;

    pvec = cross(r.direction, edge2);

    det = dot(pvec, edge1);

    if(det > -EPSILON2 && det < EPSILON2) { return false; }

    inv_det = 1.0f / det;
    tvec = r.location - tri.A;

    u = dot(tvec, pvec) * inv_det;

    if(u < 0.0f || u > 1.0f) { return false; }

    qvec = cross(tvec, edge1);

    v = dot(r.direction, qvec) * inv_det;

    if(v < 0.0f || u + v > 1.0f) { return false; }

    t = dot(edge2, qvec) * inv_det;
    intersection = r.location + t * r.direction;
    return t > EPSILON1;
}

__device__ void initial_ray(const camera& c, const point& upperleft, int x, int y, point& xgap, point& ygap, ray& r)
{
    //place the ray in the middle of the hole (not top left)
    point p = upperleft + (x + 0.5f) * xgap - (y + 0.5f) * ygap;
    r.location = p;
    r.direction = p - c.location;
    normalize(r.direction);
}



#if __GPUVERSION__
__device__
#endif
bool shootrayshared(const ray& r, int tricount, triangle *triangles, triangle& nearest, point& intersec, int threadid)
{
    float min_distance = FLT_MAX;
    bool hit;
    float distance;
    point intersection;

#if __GPUVERSION__
    const int buffersize = chunksizex * chunksizey;
    __shared__ triangle trianglebuffer[buffersize];

    int runs = (tricount + buffersize - 1) / buffersize;
    int tc = 0;
    for(int run = 0; run < runs; run++)
    {
        //sync from last run
        __syncthreads();

        //load triangles to shared memory
        if(threadid < buffersize && run * buffersize + threadid < tricount)
            { trianglebuffer[threadid] = triangles[run * buffersize + threadid]; }

        __syncthreads();
        //test for intersection
        for(int i = 0; i < buffersize && tc < tricount; tc++, i++ )
        {
            triangle t = trianglebuffer[i];
            //hit = intersect(r, t, intersection);
            hit = fastintersect(r, t, intersection);
            distance = norm(intersection - r.location);
            if(hit && distance < min_distance)
            {
                min_distance = distance;
                nearest = t;
                intersec = intersection;
            }
        }

    }
#else
    for(int i = 0; i < tricount; i++)
    {
        triangle t = triangles[i];
        //hit = intersect(r, t, intersection);
        hit = fastintersect(r, t, intersection);
        distance = norm(intersection - r.location);
        if(hit && distance < min_distance)
        {
            min_distance = distance;
            nearest = t;
            intersec = intersection;
        }
    }
#endif
    return min_distance != FLT_MAX;
}

__device__ __host__ bool shootray(const ray& r, int tricount, triangle *triangles, triangle& nearest, point& intersec)
{
    float min_distance = FLT_MAX;
    bool hit;
    float distance;
    point intersection;

    for(int i = 0; i < tricount; i++)
    {
        triangle t = triangles[i];
        //hit = intersect(r, t, intersection);
        hit = fastintersect(r, t, intersection);
        distance = norm(intersection - r.location);
        if(hit && distance < min_distance)
        {
            min_distance = distance;
            nearest = t;
            intersec = intersection;
        }
    }

    return min_distance != FLT_MAX;
}


#if __GPUVERSION__
__device__
#endif
rgb lighten(const triangle& nearest, const point& intersection, int lightcount, point *lights, int tricount, triangle *triangles)
{
    float lightintense = 0.0f;

    point normal = cross(nearest.A - nearest.B, nearest.A - nearest.C);
    normalize(normal);

    ray lightray;
    triangle lightnearest;
    point lightintersect;

    lightray.location = intersection;

    for(int i = 0; i < lightcount; i++)
    {
        point light = lights[i];
        lightray.direction = light - intersection;
        normalize(lightray.direction);

        if(!shootray(lightray, tricount, triangles, lightnearest, lightintersect) ||
                (norm(lightintersect - intersection) > norm(light - intersection)))
        {
            float cosangle = anglebetween(lightray.direction, normal);

            if(cosangle > 0)
                { lightintense += cosangle; }
        }
    }

    return shade(nearest.color, lightintense);
}

#if __CPUVERSION__
void render_pixel(rtconfig config, triangle *triangles, point *lights, rgb *resultpixels, int x, int y)
{
    int threadid = -1;
#else
__global__ void render_pixel(rtconfig config, triangle *triangles, point *lights, rgb *resultpixels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int threadid = threadIdx.y * blockDim.x + threadIdx.x;
#endif
    ray r;
    initial_ray(config.cam, config.upperleft, x, y, config.xgap, config.ygap, r);

    //find nearest intersect triangle
    point intersec;
    triangle nearest;
    if(x < config.width && y < config.height)
    {
        resultpixels[y * config.width + x] = config.background;
    }

    if(shootrayshared(r, config.tricount, triangles, nearest, intersec, threadid))
    {
        if(x < config.width && y < config.height)
        {
            //set pixel color to color of nearest intersecting triangle
            resultpixels[y * config.width + x] = lighten(nearest, intersec, config.lightcount, lights, config.tricount, triangles);
        }
    }
}

void init_ray_gap(const camera& c, int width, int height, point &xgap, point &ygap, point& upperleft)
{
    point right = cross(c.up, c.direction);
    normalize(right);

    point dx = tan(c.hor_angle * PI / 360) * c.distance * right;
    point dy = tan(c.vert_angle * PI / 360) * c.distance * c.up;

    point dir = c.direction;
    normalize(dir);
    dir = dir * c.distance;
    upperleft = c.location + dir - dx + dy ;

    xgap = dx * (2.0f / width);
    ygap = dy * (2.0f / height);
}

void render_image(const scene& s, const int& height, const int& width, rgb* image)
{
    //init config
    rtconfig config;
    config.background = s.background;
    config.cam = s.cam;
    config.width = width;
    config.height = height;

    config.tricount = s.objects.count;
    config.lightcount = s.light.count;

#if __GPUVERSION__
    cudaEvent_t start_exec, stop_exec;
    cudaEvent_t start_cpy2device, stop_cpy2device;
    cudaEvent_t start_cpy2host, stop_cpy2host;

    float time_cpy2device, time_exec, time_cpy2host, rays_per_ms;

    cudaEventCreate(&start_exec);
    cudaEventCreate(&stop_exec);
    cudaEventCreate(&start_cpy2device);
    cudaEventCreate(&stop_cpy2device);
    cudaEventCreate(&start_cpy2host);
    cudaEventCreate(&stop_cpy2host);

    cudaError_t error;

    dim3 threadsPerBlock(chunksizex, chunksizey);
    dim3 blocksPerGrid((width + chunksizex - 1) / chunksizex, (height + chunksizey - 1) / chunksizey);

    cudaEventRecord(start_cpy2device, 0);
    //copy primitives to device
    triangle *d_triangles = NULL;
    if(config.tricount > 0)
    {
        error = cudaMalloc(&d_triangles, config.tricount * sizeof(triangle));
        CHECK_EQ(cudaSuccess, error) << "Error at line " << __LINE__ << ": " << cudaGetErrorString(error);
        CHECK_NOTNULL(d_triangles);
        error = cudaMemcpyAsync(d_triangles, s.objects.triangles, config.tricount * sizeof(triangle), cudaMemcpyHostToDevice);
        CHECK_EQ(cudaSuccess, error) << "Error at line " << __LINE__ << ": " << cudaGetErrorString(error);
    }

    //copy lights to device
    point *d_lights = NULL;
    if(config.lightcount > 0)
    {
        error = cudaMalloc(&d_lights, config.lightcount * sizeof(point));
        CHECK_EQ(cudaSuccess, error) << "Error at line " << __LINE__ << ": " << cudaGetErrorString(error);
        CHECK_NOTNULL(d_lights);
        error = cudaMemcpyAsync(d_lights, s.light.lights, config.lightcount * sizeof(point), cudaMemcpyHostToDevice);
        CHECK_EQ(cudaSuccess, error) << "Error at line " << __LINE__ << ": " << cudaGetErrorString(error);
    }

    //calc ray gaps
    init_ray_gap(config.cam, config.width, config.height, config.xgap, config.ygap, config.upperleft);


    //alloc memory for result
    rgb *d_resultcolors;
    error = cudaMalloc(&d_resultcolors, width * height * sizeof(rgb));
    CHECK_EQ(cudaSuccess, error) << "Error at line " << __LINE__ << ": " << cudaGetErrorString(error);
    CHECK_NOTNULL(d_resultcolors);

    error = cudaThreadSynchronize();
    CHECK_EQ(cudaSuccess, error) << "Error at line " << __LINE__ << ": " << cudaGetErrorString(error);

    error = cudaThreadSynchronize();
    CHECK_EQ(cudaSuccess, error) << "Error at line " << __LINE__ << ": " << cudaGetErrorString(error);

    cudaEventRecord(stop_cpy2device, 0);
    cudaEventSynchronize(stop_cpy2device);

    //launch main kernel
    cudaEventRecord(start_exec, 0);
    render_pixel <<< blocksPerGrid, threadsPerBlock>>>(config, d_triangles, d_lights, d_resultcolors);
    error = cudaGetLastError();
    CHECK_EQ(cudaSuccess, error) << "Error at line " << __LINE__ << ": " << cudaGetErrorString(error);

    error = cudaThreadSynchronize();
    CHECK_EQ(cudaSuccess, error) << "Error at line " << __LINE__ << ": " << cudaGetErrorString(error);
    cudaEventRecord(stop_exec, 0);
    cudaEventSynchronize(stop_exec);

    //copy back results
    cudaEventRecord(start_cpy2host, 0);
    error = cudaMemcpy(image, d_resultcolors, width * height * sizeof(rgb), cudaMemcpyDeviceToHost);
    CHECK_EQ(cudaSuccess, error) << "Error at line " << __LINE__ << ": " << cudaGetErrorString(error);
    cudaEventRecord(stop_cpy2host, 0);
    cudaEventSynchronize(stop_cpy2host);

    cudaEventElapsedTime(&time_cpy2device, start_cpy2device, stop_cpy2device);
    cudaEventElapsedTime(&time_exec, start_exec, stop_exec);
    cudaEventElapsedTime(&time_cpy2host, start_cpy2host, stop_cpy2host);
    rays_per_ms = (width * height) / time_exec;

    std::cout << "Time to copy and alloc memory: " << time_cpy2device << " ms" << std::endl;
    std::cout << "Time to execute the kernel: " << time_exec << " ms" << std::endl;
    std::cout << "Time to copy back the result: " << time_cpy2host << " ms" << std::endl;
    std::cout << "Computed rays per millisecond: " << rays_per_ms << std::endl;

    cudaFree(d_triangles);
    cudaFree(d_lights);
    cudaFree(d_resultcolors);

    cudaEventDestroy(start_exec);
    cudaEventDestroy(stop_exec);
    cudaEventDestroy(start_cpy2device);
    cudaEventDestroy(stop_cpy2device);
    cudaEventDestroy(start_cpy2host);
    cudaEventDestroy(stop_cpy2host);
#else
    //calc ray gaps
    init_ray_gap(config.cam, config.width, config.height, config.xgap, config.ygap, config.upperleft);

    timeval start_exec, stop_exec;
    float rays_per_ms;

    gettimeofday(&start_exec, 0);
    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            render_pixel(config, s.objects.triangles, s.light.lights, image, x, y);
        }
    }
    gettimeofday(&stop_exec, 0);
    long timediff = (stop_exec.tv_sec * 1000000 + stop_exec.tv_usec - start_exec.tv_sec * 1000000 + start_exec.tv_usec) / 1000;
    rays_per_ms = (width * height) / float(timediff);
    std::cout << "Time to execute the raytracer: " << timediff << " ms" << std::endl;
    std::cout << "Computed rays per millisecond: " << rays_per_ms << std::endl;
#endif
}