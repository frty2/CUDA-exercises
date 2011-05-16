#include <cmath>
#include <glog/logging.h>
#include <iostream>

#include "float.h"
#include "raytracer.h"

#define EPSILON1 0.0001f
#define EPSILON2 0.000000001f
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
    int lightcount;
    int tricount;
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

__host__ __device__ rgb shade(const rgb& color, float fraction)
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
    if(t < EPSILON1)
    {
        return false;
    }
    intersection = location+t*direction;
    return true;
}

// checks if point p is on the same side of the line AB as C
__host__ __device__ bool inside(const point& p, const point& c, const point& a, const point& b)
{
    if( dot(cross(b-a, p-a), cross(b-a, c-a)) >= -EPSILON2 )
    {
        return true;
    }
    return false;
}

__host__ __device__ bool intersect(const ray& r, const triangle& t, point& intersection)
{
    //calc intersection with triangle surface
    point normal = cross(t.A-t.B, t.A-t.C);
    normalize(normal);
    if(!intersect(r.location, r.direction, normal, t.A, intersection))
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
    point intersection;
    for(int i = 0; i < tricount; i++)
    {
        triangle t = triangles[i];
        hit = intersect(r, t, intersection);
        distance = norm(intersection-r.location);
        if(hit && distance < min_distance)
        {
            min_distance = distance;
            nearest = t;
            intersec = intersection;
        }
    }
    return min_distance != FLT_MAX;
}

__device__ __host__ rgb lighten(const triangle& nearest, const point& intersection, int lightcount, point *lights, int tricount, triangle *triangles)
{
    float lightintense = 0.0f;
    
    point normal = cross(nearest.A-nearest.B, nearest.A-nearest.C);
    normalize(normal);
    
    ray lightray;
    triangle lightnearest;
    point lightintersect;
    
    lightray.location = intersection;
    
    for(int i = 0;i < lightcount;i++)
    {
        point light = lights[i];
        lightray.direction = light-intersection;
        normalize(lightray.direction);
        
        if(!shootray(lightray, tricount, triangles, lightnearest, lightintersect) ||
                    (norm(lightintersect - intersection) > norm(light - intersection)))
        {
            float cosangle = anglebetween(lightray.direction, normal);
            if(cosangle > 0)
                lightintense += cosangle;
        }
    }
    return shade(nearest.color, lightintense);
}

#if __CPUVERSION__
void render_pixel(rtconfig config, triangle *triangles, point *lights, rgb *resultpixels, int x, int y)
{
#else
__global__ void render_pixel(rtconfig config, triangle *triangles, point *lights, rgb *resultpixels)
{
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    int y = blockIdx.y*blockDim.y+threadIdx.y;
#endif

    if(x < config.width && y < config.height)
    {
        ray r;
        initial_ray(config.cam, config.upperleft, x, y, config.xgap, config.ygap, r);

        //find nearest intersect triangle
        point intersec;
        triangle nearest;
        
        if(shootray(r, config.tricount, triangles, nearest, intersec))
        {
            //set pixel color to color of nearest intersecting triangle
            resultpixels[y*config.width+x] = lighten(nearest, intersec, config.lightcount, lights, config.tricount, triangles);
        }
        else
        {
            resultpixels[y*config.width+x] = config.background;
        }
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
    
    config.tricount = s.objects.count;
    config.lightcount = s.light.count;
    
#if __GPUVERSION__
    cudaError_t error;

    dim3 threadsPerBlock(chunksize,chunksize);
    dim3 blocksPerGrid((width+chunksize-1)/chunksize, (height+chunksize-1)/chunksize);

    //copy primitives to device
    triangle *d_triangles = NULL;
    if(config.tricount > 0)
    {
        error = cudaMalloc(&d_triangles, config.tricount*sizeof(triangle));
        CHECK_EQ(cudaSuccess, error) << "Error at line "<< __LINE__ << ": " << cudaGetErrorString(error);
        CHECK_NOTNULL(d_triangles);
        error = cudaMemcpyAsync(d_triangles, s.objects.triangles, config.tricount*sizeof(triangle), cudaMemcpyHostToDevice);
        CHECK_EQ(cudaSuccess, error) << "Error at line "<< __LINE__ << ": " << cudaGetErrorString(error);
    }
    
    //copy lights to device
    point *d_lights = NULL;
    if(config.lightcount > 0){
        error = cudaMalloc(&d_lights, config.lightcount*sizeof(point));
        CHECK_EQ(cudaSuccess, error) << "Error at line "<< __LINE__ << ": " << cudaGetErrorString(error);
        CHECK_NOTNULL(d_lights);
        error = cudaMemcpyAsync(d_lights, s.light.lights, config.lightcount*sizeof(point), cudaMemcpyHostToDevice);
        CHECK_EQ(cudaSuccess, error) << "Error at line "<< __LINE__ << ": " << cudaGetErrorString(error);
    }
    
    //calc ray gaps
    init_ray_gap(config.cam, config.width, config.height, config.xgap, config.ygap, config.upperleft);


    //alloc memory for result
    rgb *d_resultcolors;
    error = cudaMalloc(&d_resultcolors, width*height*sizeof(rgb));
    CHECK_EQ(cudaSuccess, error) << "Error at line "<< __LINE__ << ": " << cudaGetErrorString(error);
    CHECK_NOTNULL(d_resultcolors);

    error = cudaThreadSynchronize();
    CHECK_EQ(cudaSuccess, error) << "Error at line "<< __LINE__ << ": " << cudaGetErrorString(error);

    error = cudaThreadSynchronize();
    CHECK_EQ(cudaSuccess, error) << "Error at line "<< __LINE__ << ": " << cudaGetErrorString(error);

    //launch main kernel
    render_pixel<<<blocksPerGrid, threadsPerBlock>>>(config, d_triangles, d_lights, d_resultcolors);
    error = cudaGetLastError();
    CHECK_EQ(cudaSuccess, error) << "Error at line "<< __LINE__ << ": " << cudaGetErrorString(error);
    
    error = cudaThreadSynchronize();
    CHECK_EQ(cudaSuccess, error) << "Error at line "<< __LINE__ << ": " << cudaGetErrorString(error);

    //copy back results
    error = cudaMemcpy(image, d_resultcolors, width*height*sizeof(rgb), cudaMemcpyDeviceToHost);
    CHECK_EQ(cudaSuccess, error) << "Error at line "<< __LINE__ << ": " << cudaGetErrorString(error);

    cudaFree(d_triangles);
    cudaFree(d_lights);
    cudaFree(d_resultcolors);
#else
    //calc ray gaps
    init_ray_gap(config.cam, config.width, config.height, config.xgap, config.ygap, config.upperleft);

    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            render_pixel(config, s.objects.triangles, s.light.lights, image, x, y);
        }
    }
#endif
}

