#pragma once

#include <vector>
#include <iostream>
#include "vector_types.h"

typedef float3 point;

typedef uchar4 rgb;

struct triangle
{
    point A;
    point B;
    point C;
    rgb color;
};

struct primitives
{
    triangle *triangles;
    int count;
};

struct lighting
{
    point *lights;
    int count;
};

struct camera
{
    point location;
    point direction;
    point up;
    float distance;
    float hor_angle;
    float vert_angle;
};

struct ray
{
    point location;
    point direction;
};

struct scene
{
    rgb background;
    primitives objects;
    lighting light;
    camera cam;
};

// define types
typedef struct triangle triangle;
typedef struct primitives primitives;
typedef struct camera camera;
typedef struct ray ray;
typedef struct scene scene;

// define stream output for types
std::ostream& operator <<(std::ostream& s, const point& p);
std::ostream& operator <<(std::ostream& s, const rgb& r);
std::ostream& operator <<(std::ostream& s, const triangle& t);
std::ostream& operator <<(std::ostream& s, const camera& c);
std::ostream& operator <<(std::ostream& s, const ray& r);

