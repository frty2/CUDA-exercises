#pragma once

#include <vector>
#include <iostream>

// define structs
struct point
{
    double x;
    double y;
    double z;
};

struct rgb
{
    int red;
    int green;
    int blue;
};

struct triangle
{
    point A;
    point B;
    point C;
    point norm;
    rgb color;
};

struct primitives
{
    int count;
    triangle *triangles;
};

struct camera
{
    point location;
    point direction;
    point up;
    point upperleft;
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
    camera cam;
};

// define types
typedef struct point point;
typedef struct rgb rgb;
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

