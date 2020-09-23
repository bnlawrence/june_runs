#include <time.h>
#include "get_time.h"

/* Get wall clock time in seconds as a double */
double get_time()
{
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return ((double) t.tv_nsec)/1.0e9+t.tv_sec;
}
