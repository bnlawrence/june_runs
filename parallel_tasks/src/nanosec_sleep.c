#include <time.h>
#include "nanosec_sleep.h"

/* Sleep for specified number of nanoseconds */
void nanosec_sleep(long long n)
{
  struct timespec ts;
  ts.tv_sec  = n / 1000000000;
  ts.tv_nsec = n % 1000000000;
  nanosleep(&ts, NULL);
}
