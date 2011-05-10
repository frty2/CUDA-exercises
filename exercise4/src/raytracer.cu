#include <cmath>
#include <glog/logging.h>
#include <iostream>

#include "float.h"
#include "raytracer.h"

#define EPSILON 0
#define PI 3.14159265358f

#define chunksize 8

struct rtconfig
{
    rgb background;
    camera cam;
    int width;
    int height;
    point xgap;
    point ygap;
    point upperleft;
};
typedef struct rtconfig rtconfig;


__host__ __device__ point cross(const point& p1, const point& p2)
{
    point result;
    result.x = p1.y*p2.z - p1.z*p2.y;
    result.y = p1.z*p2.x - p1.x*p2.z;
    result.z = p1.x*p2.y - p1.y*p2.x;
    return result;
}

__host__ __device__ float dot(const point& p1, const point& p2)
{
    return p1.x*p2.x + p1.y*p2.y + p1.z*p2.z;
}

__host__ __device__ float norm(const point& p)
{
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}

__host__ __device__ float anglebetween(const point& p1, const point& p2)
{
    return dot(p1, p2)/(norm(p1)*norm(p2));
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
    result.x = scalar*vector.x;
    result.y = scalar*vector.y;
    result.z = scalar*vector.z;
    return result;
}

__host__ __device__ point operator*(float scalar, const point& vector)
{
    return vector * scalar;
}

__host__ __device__ point operator+(const point& left, const point& right)
{
    point result;
    result.x = left.x+right.x;
    result.y = left.y+right.y;
    result.z = left.z+right.z;

    return result;
}

__host__ __device__ point operator-(const point& left, const point& right)
{
    point result;
    result.x = left.x-right.x;
    result.y = left.y-right.y;
    result.z = left.z-right.z;

    return result;
}

__device__ rgb shade(const rgb& color, float fraction)
{
    if(fraction < 0.0f) fraction = 0.0f;
    if(fraction > 1.0f) fraction = 1.0f;
    rgb result;
    result.x = color.x*fraction;
    result.z = color.z*fraction;
    result.y = color.y*fraction;
    return result;
}

__host__ __device__ bool intersect(const point& location, const point& direction, const point& normal, const point& p, point& intersection)
{
    float t = dot(normal, p-location) / dot(normal,direction);

    //wrong direction
    if(t < 0.0f)
    {
        return false;
    }
    intersection = location+t*direction;
    return true;
}

// checks if point p is on the same side of the line AB as C
__host__ __device__ bool inside(const point& p, const point& c, const point& a, const point& b)
{
    if( dot(cross(b-a, p-a), cross(b-a, c-a)) >= -EPSILON )
    {
        return true;
    }
    return false;
}

__host__ __device__ bool intersect(const ray& r, const triangle& t, point& intersection)
{
    //calc intersection with triangle surface
    if(!intersect(r.location, r.direction, t.norm, t.A, intersection))
    {
        return false;
    }

    //check if intersection is within triangle
    if(inside(intersection, t.A, t.B, t.C) && inside(intersection, t.B, t.A, t.C) && inside(intersection, t.C, t.A, t.B) )
    {
        return true;
    }
    return false;
}

__device__ void initial_ray(const camera& c, const point& upperleft, int x, int y, point& xgap, point& ygap, ray& r)
{
    //place the ray in the middle of the hole (not top left)
    point p = upperleft + (x+0.5f) * xgap - (y+0.5f) * ygap;
    r.location = p;
    r.direction = p-c.location;
    normalize(r.direction);
}

__device__ __host__ bool shootray(const ray& r, int tricount, triangle *triangles, triangle& nearest, point& intersec)
{
    float min_distance = FLT_MAX;
    bool hit;
    float distance;
    for(int i = 0; i < tricount; i++)
    {
        hit = intersect(r, triangles[i], intersec);
        distance = norm(intersec-r.location);
        if(hit && distance < min_distance && distance >= -EPSILON)
        {
            nearest = triangles[i];
            min_distance = distance;
        }
    }
	return min_distance != FLT_MAX;
}

#if __CPUVERSION__
void render_pixel(rtconfig *config, int tricount, triangle *triangles, int lightcount, point *lights, rgb *resultpixels, int x, int y)
{
#else
__global__ void render_pixel(rtconfig *config, int tricount, triangle *triangles, int lightcount, point *lights, rgb *resultpixels)
{
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;
#endif

    if(x < config->width && y < config->height)
    {
		ray r;
        initial_ray(config->cam, config->upperleft, x, y, config->xgap, config->ygap, r);

        //find nearest intersect triangle
        point intersec;
        triangle nearest;
        resultpixels[y*config->width+x] = config->background;
        
        if(shootray(r, tricount,triangles, nearest, intersec))
        {
            //set pixel color to color of nearest intersecting triangle
            float angle = anglebetween(nearest.norm, r.direction);
            float lightintense = fabs(angle);
            resultpixels[y*config->width+x] = shade(nearest.color, lightintense);
        }
    }
}

//calculates the norm for every triangle

__global__ void init_norms(int count, triangle *triangles)
{
#if __GPUVERSION__
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if(tid < count)
#else
    for(int tid = 0; tid < count; tid++)
#endif
    {
        triangle t = triangles[tid];
        t.norm = cross(t.A - t.C, t.A - t.B);
        normalize( t.norm );
        triangles[tid].norm = t.norm;
    }
}


void init_ray_gap(const camera& c, int width, int height, point &xgap, point &ygap, point& upperleft)
{
    point right = cross(c.up, c.direction);
    normalize(right);

    point dx = tan(c.hor_angle*PI/360) * c.distance * right;
    point dy = tan(c.vert_angle*PI/360) * c.distance * c.up;

    point dir = c.direction;
    normalize(dir);
    dir = dir*c.distance;
    upperleft = c.location + dir - dx + dy ;

    xgap = dx*(2.0f/width);
    ygap = dy*(2.0f/height);
}

void render_image(const scene& s, const int& height, const int& width, rgb* image)
{
    //init config
    rtconfig config;
    config.background = s.background;
    config.cam = s.cam;
    config.width = width;
    config.height = height;
    
    int tricount = s.objects.count;
    int lightcount = s.light.count;
    
#if __GPUVERSION__
    cudaError_t error;

    dim3 threadsPerBlock(chunksize,chunksize);
    dim3 blocksPerGrid((width+chunksize-1)/chunksize, (height+chunksize-1)/chunksize);

    //copy primitives to device
    triangle *d_triangles = NULL;
    if(tricount > 0)
    {
        error = cudaMalloc(&d_triangles, tricount*sizeof(triangle));
        CHECK_NOTNULL(d_triangles);
        CHECK_EQ(cudaSuccess, error) << "Error at line "<< __LINE__ << ": " << cudaGetErrorString(error);
        error = cudaMemcpyAsync(d_triangles, s.objects.triangles, tricount*sizeof(triangle), cudaMemcpyHostToDevice);
        CHECK_EQ(cudaSuccess, error) << "Error at line "<< __LINE__ << ": " << cudaGetErrorString(error);
    }
    
    //copy lights to device
    point *d_lights = NULL;
    if(lightcount > 0){
        error = cudaMalloc(&d_lights, lightcount*sizeof(point));
        CHECK_NOTNULL(d_lights) ;
        CHECK_EQ(cudaSuccess, error) << "Error at line "<< __LINE__ << ": " << cudaGetErrorString(error);
        error = cudaMemcpyAsync(d_lights, s.light.lights, lightcount*sizeof(point), cudaMemcpyHostToDevice);
        CHECK_EQ(cudaSuccess, error) << "Error at line "<< __LINE__ << ": " << cudaGetErrorString(error);
    }
    
    //calc ray gaps
    init_ray_gap(config.cam, config.width, config.height, config.xgap, config.ygap, config.upperleft);

    //copy config to device
    int csize = sizeof(rtconfig);
    rtconfig *d_config;
    error = cudaMalloc(&d_config, csize);
    CHECK_NOTNULL(d_config);
    CHECK_EQ(cudaSuccess, error) << "Error at line "<< __LINE__ << ": " << cudaGetErrorString(error);
    error = cudaMemcpyAsync(d_config, &config, csize, cudaMemcpyHostToDevice);
    CHECK_EQ(cudaSuccess, error) << "Error at line "<< __LINE__ << ": " << cudaGetErrorString(error);

    //alloc memory for result
    rgb *d_resultcolors;
    cudaMalloc(&d_resultcolors, width*height*sizeof(rgb));
    CHECK_NOTNULL(d_resultcolors);// << "Error at line "<< __LINE__ << ": Not enough memory for result image";
    error = cudaGetLastError();
    CHECK_EQ(cudaSuccess, error) << "Error at line "<< __LINE__ << ": " << cudaGetErrorString(error);

    error = cudaThreadSynchronize();
    CHECK_EQ(cudaSuccess, error) << "Error at line "<< __LINE__ << ": " << cudaGetErrorString(error);

    //calc primitives norms
    int n = 512;
    dim3 normThreadsPerBlock(n);
    dim3 normBlocksPerGrid((tricount + n - 1) / n);
    init_norms<<<normBlocksPerGrid, normThreadsPerBlock>>>(tricount, d_triangles);
    error = cudaGetLastError();
    CHECK_EQ(cudaSuccess, error) << "Error at line "<< __LINE__ << ": " << cudaGetErrorString(error);

    error = cudaThreadSynchronize();
    CHECK_EQ(cudaSuccess, error) << "Error at line "<< __LINE__ << ": " << cudaGetErrorString(error);

    //launch main kernel
    render_pixel<<<blocksPerGrid, threadsPerBlock>>>(d_config, tricount, d_triangles, lightcount, d_lights, d_resultcolors);
    error = cudaGetLastError();
    CHECK_EQ(cudaSuccess, error) << "Error at line "<< __LINE__ << ": " << cudaGetErrorString(error);

    //copy back results
    error = cudaMemcpy(image, d_resultcolors, width*height*sizeof(rgb), cudaMemcpyDeviceToHost);
    CHECK_EQ(cudaSuccess, error) << "Error at line "<< __LINE__ << ": " << cudaGetErrorString(error);

    cudaFree(d_triangles);
    cudaFree(d_lights);
    cudaFree(d_config);
    cudaFree(d_resultcolors);
#else
    //calc ray gaps
    init_norms(tricount, s.objects.triangles);
    init_ray_gap(config.cam, config.width, config.height, config.xgap, config.ygap, config.upperleft);

    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            render_pixel(&config, tricount, s.objects.triangles, lightcount, s.light.lights, image, x, y);
        }
    }
#endif
}

