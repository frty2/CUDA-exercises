#include <glog/logging.h>
#include <fstream>
#include <iterator>
#include <string>
#include <string>
#include <iostream>
#include <map>
#include <stdlib.h>

#include "parser.h"
#include "raytracer.h"

void operator >>(const YAML::Node& node, point& v)
{
    node[0] >> v.x;
    node[1] >> v.y;
    node[2] >> v.z;
}

void operator >>(const YAML::Node& node, rgb& r)
{
    int rt, gt, bt;

    node[0] >> rt;
    r.x = rt;
    node[1] >> gt;
    r.y = gt;
    node[2] >> bt;
    r.z = bt;
}

void operator >>(const YAML::Node& node, triangle& t)
{
    node[0][0] >> t.A.x;
    node[0][1] >> t.A.y;
    node[0][2] >> t.A.z;

    node[1][0] >> t.B.x;
    node[1][1] >> t.B.y;
    node[1][2] >> t.B.z;

    node[2][0] >> t.C.x;
    node[2][1] >> t.C.y;
    node[2][2] >> t.C.z;
}

void operator >>(const YAML::Node& node, primitives& p)
{
    int size = node.size();
    p.triangles = (triangle*) malloc( size * sizeof(triangle) );
    p.count = size;

    for(int i = 0; i < size; i++)
    {
        node[i]["triangle"] >> p.triangles[i];
        node[i]["color"] >> p.triangles[i].color;
    }
}

void operator >>(const YAML::Node& node, lighting& l)
{
    int size = node.size();
    l.lights = (point*) malloc( size * sizeof(point) );
    l.count = size;
    for(int i = 0; i < size; i++)
    {
        node[i] >> l.lights[i];
    }
}

void operator >>(const YAML::Node& node, camera& c)
{
    if(const YAML::Node *pValue = node.FindValue("location"))
    {
        *pValue >> c.location;
    }
    if(const YAML::Node *pValue = node.FindValue("direction"))
    {
        *pValue >> c.direction;
        normalize(c.direction);
    }
    if(const YAML::Node *pValue = node.FindValue("up"))
    {
        *pValue >> c.up;
        normalize(c.up);
    }
    if(const YAML::Node *pValue = node.FindValue("distance"))
    {
        *pValue >> c.distance;
    }
    if(const YAML::Node *pValue = node.FindValue("horizontal_angle"))
    {
        *pValue >> c.hor_angle;
    }
    if(const YAML::Node *pValue = node.FindValue("vertical_angle"))
    {
        *pValue >> c.vert_angle;
    }
}

void find_primitives(const YAML::Node& node, primitives& p)
{
    if(const YAML::Node *pValue = node.FindValue("primitives"))
    {
        *pValue >> p;
    }
    else
    {
        p.count = 0;
    }
}

void find_camera(const YAML::Node& node, camera& c)
{
    if(const YAML::Node *pValue = node.FindValue("camera"))
    {
        *pValue >> c;
    }
    else
    {
        std::cout << "Warning: no camera defined, set default" << std::endl;
        point location, direction, up;
        location.x = 0;
        location.y = 0;
        location.z = 0;
        direction.x = 0;
        direction.y = 0;
        direction.z = 1;
        up.x = 0;
        up.y = 1;
        up.z = 0;
        c.location = location;
        c.direction = direction;
        c.up = up;
        c.distance = 1;
        c.hor_angle = 90;
        c.vert_angle = 90;
    }
}

void find_background(const YAML::Node& node, rgb& b)
{
    if(const YAML::Node *pValue = node.FindValue("background"))
    {
        *pValue >> b;
    }
    else
    {
        std::cout << "Warning: no background color defined, set default to black" << std::endl;
        b.x = 0;
        b.y = 0;
        b.z = 0;
    }
}

void find_lights(const YAML::Node& node, lighting& l)
{
    if(const YAML::Node *pValue = node.FindValue("lights"))
    {
        *pValue >> l;
    }
    else
    {
        l.count = 0;
    }
}

void parse_scene(const char* filename, scene& s)
{
    std::ifstream fin(filename);
    YAML::Parser parser(fin);
    YAML::Node doc;
    while(parser.GetNextDocument(doc))
    {
        find_primitives(doc, s.objects);
        find_camera(doc, s.cam);
        find_background(doc, s.background);
        find_lights(doc, s.light);
    }
    fin.close();
}
