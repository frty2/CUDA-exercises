//#define GOOGLE_STRIP_LOG 2
#include <glog/logging.h>
#include <gflags/gflags.h>
#include <stdlib.h>
#include <fstream>

#include "types.h"
#include "parser.h"
#include "raytracer.h"
#include "ppm_writer.h"

#if __GPUVERSION__
#include "copyprimitives.h"
#endif

static bool validateWidthAndHeight(const char* flagname, int value)
{
    if (value > 0 && value < 32768)
        return true;
    printf("Invalid value for --%s: %d\n", flagname, (int)value);
    return false;
}

DEFINE_int32(width, 0, "width of the rendered scene in pixels");
DEFINE_int32(height, 0, "height of the rendered scene in pixels");
static const bool width_dummy = google::RegisterFlagValidator(&FLAGS_width, &validateWidthAndHeight);
static const bool height_dummy = google::RegisterFlagValidator(&FLAGS_height, &validateWidthAndHeight);

int main(int argc, char **argv)
{
    // logging
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    // flags
    google::ParseCommandLineFlags(&argc, &argv, true);

    // checking command line arguments
    if (argc != 3)
    {
        std::cerr << "You have to specify a scene file and a output file" << std::endl;
        return -1;
    }

    CHECK_STRNE(argv[1], "") << "No scene file specified.";
    CHECK_STRNE(argv[2], "") << "No output file specified.";

    // parse scene
    scene s;
    parse_scene(argv[1], s);

    // this is our height and width
    int width = FLAGS_width;
    int height = FLAGS_height;

    // changed to dynamic array to get more pixels
    rgb *image;

    image = (rgb*) malloc(width*height*sizeof(rgb));
    if(! image)
    {
        std::cout << "Not enough memory for image" << std::endl;
        return -1;
    }
#if __GPUVERSION__
    copyPrimitives(s.objects);
#endif
    // render the scene
    render_image(s, height, width, image);

    // write image to filename
    write_ppm(image, width, height, argv[2]);


    return 0;
}
