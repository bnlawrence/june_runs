#include <pthread.h>

#include "job_info.h"

/* Number of processors, ID of this processor */
int NTask;
int ThisTask;

/* Flag to signal job completion */
pthread_mutex_t job_running_mutex = PTHREAD_MUTEX_INITIALIZER;
int job_running = 0;

/* Maximum length of command to execute */
char command[COMMAND_LENGTH];

/* Range of job indexes to do */
int ifirst, ilast;

/* Table of job results */
int *job_result;

/* Job elapsed times */
double *job_time;

/* Lines read from command file */
char **command_line;

/* Whether we have the command file */
int have_command_file;
