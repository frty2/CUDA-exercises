#include "types.h"

std::ostream& operator <<(std::ostream& s, const point& p)
{
    s << "Point:  x:" << p.x << ", y:" << p.y << ", z:" << p.z << std::endl;
    return s;
}

std::ostream& operator <<(std::ostream& s, const rgb& r)
{
    s << "Color:  r:" << int(r.x) << ", g:" << int(r.y) << ", b:" << int(r.z) << std::endl;
    return s;
}

std::ostream& operator <<(std::ostream& s, const triangle& t)
{
    s << "Triangle:" << std::endl << "# " << t.A << std::endl << "# " << t.B << std::endl << "# " << t.C << std::endl << "# " << t.color;
    return s;
}

std::ostream& operator <<(std::ostream& s, const camera& c)
{
    s << "Camera: " << std::endl << "# Location " << c.location << "# Direction " << c.direction << "# Up " << c.up << "# Hor_angle: " << c.hor_angle << std::endl << "# Vert_angle: " << c.vert_angle << std::endl << "# Distance: " << c.distance << std::endl;
    return s;
}

std::ostream& operator <<(std::ostream& s, const ray& r)
{
    s << "Ray: " << std::endl << "\t-> " << r.location << std::endl << "\t-> " << r.direction << std::endl;
    return s;
}