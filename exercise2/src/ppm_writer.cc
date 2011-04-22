#include "ppm_writer.h"

#include <iostream>
#include <fstream>

void write_ppm(rgb* pixelarray, int width, int height, char* filename)
{
    std::ofstream file (filename);
    file << "P3" << std::endl;
    file << width << " " << height << std::endl;
    file << "255" << std::endl;
    int i = 0;
    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            int pos = y*width+x;
            //newline after 5 pixel to match max 70 chars per line rule
            //(3 chars per color + 1 spacing) * 3 colors * 5 pixel = 60 pixel
            if(++i % 5 == 0)
            {
                file << pixelarray[pos].red << " " << pixelarray[pos].green << " " << pixelarray[pos].blue << std::endl;
            }
            else
            {
                file << pixelarray[pos].red << " " << pixelarray[pos].green << " " << pixelarray[pos].blue << " ";
            }
        }
    }
    file.close();
}

