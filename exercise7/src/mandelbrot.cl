__kernel void mandelbrot (__global char* output)
{
    int width = get_global_size(0);
    int height = get_global_size(1);
    int threadx = get_global_id(0);
    int thready = get_global_id(1);
    
    float cx = (threadx * 4.0f / width ) - 2.0f;
    float cy = (thready * 4.0f / height ) - 2.0f;
    
    float x = 0.0f;
    float y = 0.0f;
    
    float xn, yn;
    int i;
    for(i = 0;i < 100 && (x*x+y*y) <= 4.0f;i++)
    {
        xn = x * x - y * y + cx;
        yn = 2 * x * y + cy;
        x = xn;
        y = yn;
    }
    output[ thready*width+threadx ] = i;
}