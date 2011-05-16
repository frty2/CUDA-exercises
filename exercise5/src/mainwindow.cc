#include "mainwindow.h"

#if __APPLE__
    #include <GLUT/glut.h>
    #include <OpenGL/gl.h>
#else
    #include <GL/gl.h>
    #include <GL/glut.h>
#endif

#include <stdlib.h>
#include <glog/logging.h>

#define ESC 27

int width;
int height;
float* pixels;

void paint(void)
{ 
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glDrawPixels(width, height, GL_RGB, GL_FLOAT, pixels);

    glutSwapBuffers();
} 

void keypressed(unsigned char key, int x, int y)
{
    if(key == ESC)
    {
        exit(0);
    }
}
void displayimage(int argc, char **argv, rgb* pixelarray, int w, int h)
{
    width = w;
    height = h;
    pixels = (float*) malloc(3*width*height*sizeof(float));
    CHECK_NOTNULL(pixels);
    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            pixels[3*(y*width+x)] = pixelarray[(height-y-1)*width+x].x/255.0f;
            pixels[3*(y*width+x)+1] = (float)pixelarray[(height-y-1)*width+x].y/255.0f;
            pixels[3*(y*width+x)+2] = (float)pixelarray[(height-y-1)*width+x].z/255.0f;
        }
    }
    glutInit(&argc, argv); 
    glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH); 
    glutInitWindowPosition(100,100);
    glutInitWindowSize(width,height);
    glutCreateWindow (argv[0]);
    glClearColor(0.0, 0.0, 0.0, 0.0);
        
    glutDisplayFunc(paint);
    glutKeyboardFunc(keypressed);
    //glutMouseFunc(mousepressed);
    glutIdleFunc(NULL);
    glutMainLoop();
}