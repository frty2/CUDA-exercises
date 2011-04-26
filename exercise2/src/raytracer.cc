#include <glog/logging.h>
#include <cmath>
#include "float.h"
#include "raytracer.h"
#include <iostream>

#define EPSILON 5e-4

point cross(const point& p1, const point& p2)
{
    point result;
    result.x = p1.y*p2.z - p1.z*p2.y;
    result.y = p1.z*p2.x - p1.x*p2.z;
    result.z = p1.x*p2.y - p1.y*p2.x;
    return result;
}

float dot(const point& p1, const point& p2)
{
    return p1.x*p2.x + p1.y*p2.y + p1.z*p2.z;
}

float norm(const point& p)
{
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}

void normalize(point& p)
{
    double len = norm(p);
    if(len != 0)
    {
        p.x /= len;
        p.y /= len;
        p.z /= len;
    }
}

point operator*(const point& vector, double scalar)
{
    point result;
    result.x = scalar*vector.x;
    result.y = scalar*vector.y;
    result.z = scalar*vector.z;
    return result;
}

point operator*(double scalar, const point& vector)
{
    return vector * scalar;
}

point operator+(const point& left, const point& right)
{
    point result;
    result.x = left.x+right.x;
    result.y = left.y+right.y;
    result.z = left.z+right.z;

    return result;
}

point operator-(const point& left, const point& right)
{
    point result;
    result.x = left.x-right.x;
    result.y = left.y-right.y;
    result.z = left.z-right.z;

    return result;
}

bool intersect(const point& location, const point& direction, const point& normal, const point& p, point& intersection)
{
    double t = dot(normal, p-location) / dot(normal,direction);

    //wrong direction
    if(t < 0)
    {
        return false;
    }

    intersection = location+t*direction;
    return true;
}

// checks if point p is on the same side of the line AB as C
bool inside(const point& p, const point& c, const point& a, const point& b)
{
    point side1 = cross(b-a, p-a);
    point side2 = cross(b-a, c-a);

    if( dot(side1, side2) >= -EPSILON )
    {
        return true;
    }
    return false;
}

bool intersect(const ray& r, const triangle& t, point& intersection)
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

void initial_ray(const camera& c, int x, int y, point& xgap, point& ygap, ray& r)
{
    //place the ray in the middle of the hole (not top left)
    point p = c.upperleft + (x+0.5) * xgap - (y+0.5) * ygap;
    r.location = p;
    r.direction = p-c.location;
    normalize(r.direction);
}

void init_ray_gap(const camera& c, int width, int height, point &xgap, point &ygap)
{
    point pic_center;
    intersect(c.location, c.direction, c.direction , c.upperleft, pic_center);

    point right = cross(c.up, c.direction);
    normalize(right);

    point pic_center_top;


    point down = -1*c.up;
    normalize(down);

    intersect(c.upperleft, right, right, pic_center, pic_center_top);

    xgap = (pic_center_top-c.upperleft)*(2.0/width);
    ygap = (pic_center_top-pic_center)*(2.0/height);
}

//calculates the norm for every triangle
void init_norms(const scene& s)
{
    for(int i = 0; i < s.objects.count; i++)
    {
        triangle t = s.objects.triangles[i];
        t.norm = cross(t.A - t.C, t.A - t.B);
        normalize( t.norm );
        s.objects.triangles[i].norm = t.norm;
    }
}

void render_image(const scene& s, const int& height, const int& width, rgb* image)
{
    init_norms(s);

    point xgap, ygap;
    init_ray_gap(s.cam, width, height, xgap, ygap);

    int x,y;
    for(y = 0; y < height; y++)
    {
        for(x = 0; x < width; x++)
        {
            ray r;
            initial_ray(s.cam, x, y, xgap, ygap, r);

			std::cout << r;
			
            //find nearest intersect triangle
            triangle nearest;
            double max_distance = DBL_MAX;
            int i;
            for(i = 0; i < s.objects.count; i++)
            {
                point intersec;
                bool hit = intersect(r, s.objects.triangles[i], intersec);
                float distance = norm(s.cam.location - intersec);
                if(hit && distance < max_distance && distance > 0)
                {
                    nearest = s.objects.triangles[i];
                    max_distance = distance;
                }
            }


            if(max_distance == DBL_MAX)
            {
                //no intersecting triangle, set color to background
                image[y*width+x] = s.background;
            }
            else
            {
                //set pixel color to color of nearest intersecting triangle
                image[y*width+x] = nearest.color;
            }

        }
    }
}

