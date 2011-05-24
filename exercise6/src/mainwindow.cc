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
rgb* pixels;

void paint(void)
{
    glClear(GL_COLOR_BUFFER_BIT);
    glWindowPos2i(0, height);
    glPixelZoom(1.0, -1.0);
    glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

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
    pixels = pixelarray;
    glutInit(&argc, argv);
    glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowPosition(100, 100);
    glutInitWindowSize(width, height);
    glutCreateWindow (argv[0]);
    glClearColor(0.0, 0.0, 0.0, 0.0);

    glutDisplayFunc(paint);
    glutKeyboardFunc(keypressed);
    //glutMouseFunc(mousepressed);
    glutIdleFunc(NULL);
    glutMainLoop();
}