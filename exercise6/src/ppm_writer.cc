#include "ppm_writer.h"

#include <iostream>
#include <fstream>

void write_ppm(rgb* pixelarray, int width, int height, char* filename)
{
    std::ofstream file (filename);
    file << "P3" << std::endl;
    file << width << " " << height << std::endl;
    file << "255" << std::endl;
    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            int pos = y * width + x;
            file << (int)pixelarray[pos].x << " " << (int)pixelarray[pos].y << " " << (int)pixelarray[pos].z << std::endl;
        }
    }
    file.close();
}

