#include <mpi.h>

#include "worker_task.h"
#include "run_job.h"
#include "job_info.h"
#include "nanosec_sleep.h"

void worker_task(void)
{
  /*
    Processors other than 0 request and execute jobs
    until there are none left (signalled by -ve job ID).

    To request a job we send the index of the last job we
    completed (or -1 if there was no job) to rank 0.
  */
  int last_job = -1;

  MPI_Barrier(MPI_COMM_WORLD);
  while(1)
    {
      int ijob;
      /* Ask for a job */
      MPI_Sendrecv(&last_job, 1, MPI_INT, 0, JOB_REQUEST_TAG,
                   &ijob,     1, MPI_INT, 0, JOB_RESPONSE_TAG,
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if(ijob >= 0)
        {
          /* Run the job if we got one */
          run_job(&ijob);
          last_job = ijob;
        }
      else
        {
          /* If there are no jobs left, we're done */
          last_job = -1;
          break;
        }
    }
#ifdef TERMINATE_SIGNAL
  /*
    Now wait for termination signal from task 0
  */
  int dummy;
  MPI_Request termination_request;
  MPI_Irecv(&dummy, 1, MPI_INT, 0, TERMINATION_TAG, 
            MPI_COMM_WORLD, &termination_request);
  long long sleep_nsecs = 1;
  while(1)
    {
      /* Check if we got the terminate signal */
      int flag;
      MPI_Test(&termination_request, &flag, MPI_STATUS_IGNORE);
      if(flag)break;
          
      /* If we didn't, wait a bit before trying again */
      nanosec_sleep(sleep_nsecs);
      if(sleep_nsecs < MAX_SLEEP_NS)sleep_nsecs *= 2;
    }
#endif
}
