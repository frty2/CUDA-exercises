#include "types.h"

std::ostream& operator <<(std::ostream& s, const point& p)
{
    s << "Point:  x:" << p.x << ", y:" << p.y << ", z:" << p.z << std::endl;
    return s;
}

std::ostream& operator <<(std::ostream& s, const rgb& r)
{
    s << "Color:  r:" << r.red << ", g:" << r.green << ", b:" << r.blue<< std::endl;
    return s;
}

std::ostream& operator <<(std::ostream& s, const triangle& t)
{
    s << "Triangle:" << std::endl << "# " << t.A << std::endl << "# " << t.B << std::endl << "# " << t.C << std::endl << "# " << t.color;
    return s;
}

std::ostream& operator <<(std::ostream& s, const camera& c)
{
    // TODO
    // fill me
    return s;
}

std::ostream& operator <<(std::ostream& s, const ray& r)
{
    s << "Ray: " << std::endl << "\t-> " << r.location << std::endl << "\t-> " << r.direction << std::endl;
    return s;
}