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
    node[0] >> r.red;
    node[1] >> r.green;
    node[2] >> r.blue;
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
    p.triangles = (triangle*) malloc( size*sizeof(triangle) );
    p.count = size;

    for(int i=0; i < size; i++)
    {
        node[i]["triangle"] >> p.triangles[i];
        node[i]["color"] >> p.triangles[i].color;
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
    }
    if(const YAML::Node *pValue = node.FindValue("upper_left"))
    {
        *pValue >> c.upperleft;
    }
}

void parse_scene(const char* filename, scene& s)
{
    std::ifstream fin(filename);
    YAML::Parser parser(fin);
    YAML::Node doc;
    //std::cout << doc << std::endl;
    parser.GetNextDocument(doc);

    find_primitives(doc, s.objects);
    find_camera(doc, s.cam);
    find_background(doc, s.background);
}

void find_primitives(const YAML::Node& node, primitives& p)
{
    if(const YAML::Node *pValue = node.FindValue("primitives"))
    {
        *pValue >> p;
    }
}

void find_camera(const YAML::Node& node, camera& c)
{
    if(const YAML::Node *pValue = node.FindValue("camera"))
    {
        *pValue >> c;
    }

}

void find_background(const YAML::Node& node, rgb& b)
{
    if(const YAML::Node *pValue = node.FindValue("background"))
    {
        *pValue >> b;
    }
}
