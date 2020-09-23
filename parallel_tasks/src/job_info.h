#ifndef JOB_INFO_H
#define JOB_INFO_H

#include <pthread.h>

/* Number of processors, ID of this processor */
extern int NTask;
extern int ThisTask;

/* Flag to signal job completion */
extern pthread_mutex_t job_running_mutex;
extern int job_running;

/* Maximum length of command to execute */
#define COMMAND_LENGTH 4096
extern char command[COMMAND_LENGTH];

/* MPI tags */
#define JOB_REQUEST_TAG  1
#define JOB_RESPONSE_TAG 2
#define TERMINATION_TAG  3

/* Range of job indexes to do */
extern int ifirst, ilast;

/* Table of job results */
extern int *job_result;

/* Job elapsed times */
extern double *job_time;

#define TERMINATE_SIGNAL

/* Maximum time to wait when polling for incoming messages (nanosecs) */
#define MAX_SLEEP_NS 1000000000

/* Whether we have the command file */
extern int have_command_file;

/* Lines read from command file */
extern char **command_line;

#endif
