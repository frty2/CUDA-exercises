#pragma once

#include "types.h"

// vector operations
point cross(const point& p1, const point& p2);
float dot(const point& p1, const point& p2);
float norm(const point& p);
float anglebetween(const point& p1, const point& p2);

void normalize(point& p);
point operator+(const point& left, const point& right);
point operator*(const point& vector, float scalar);
point operator*(float scalar, const point& vector);

rgb shade(const rgb& color, float fraction);


// intersection
bool intersect(const ray& r, const triangle& t, point& intersection);

// initial rays
void initial_ray(const camera& c, int x, int y, point& xgap, point& ygap, ray& r);



extern "C" void render_image(const scene& s, const int& height, const int& width, rgb* image);

