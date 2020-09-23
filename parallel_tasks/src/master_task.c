#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <pthread.h>

#include "master_task.h"
#include "nanosec_sleep.h"
#include "job_info.h"
#include "run_job.h"
#include "terminate.h"

void master_task(void)
{
  /*
    Processor 0 needs to handle job requests and 
    run its own jobs at the same time. Jobs on this
    process are run in a separate thread so we can continue
    responding to job requests. The main thread spends most
    of its time sleeping to avoid taking cpu time from the jobs.
  */

  /* Initial number of jobs to assign */
  int next_to_assign = ifirst;
  int last_job = -1;

  /* Need to track which processors have finished */
  int nfinished  = 0;
  int proc0_done = 0;
  int local_job  = -1;

  /* Initially have no local job running */
  pthread_t job_thread;
  pthread_mutex_lock( &job_running_mutex);
  job_running = 0;
  pthread_mutex_unlock( &job_running_mutex);

  /* Post receives from all other tasks */
  int src;
  MPI_Request *requests = malloc(sizeof(MPI_Request)*(NTask-1));
  int *recvbufs = malloc(sizeof(int)*(NTask-1));
  int *index_complete = malloc(sizeof(int)*(NTask-1));
  for(src=1;src<NTask;src+=1)
    {
      MPI_Irecv(&recvbufs[src-1], 1, MPI_INT, src, JOB_REQUEST_TAG, 
                MPI_COMM_WORLD, &requests[src-1]);
    }
  MPI_Barrier(MPI_COMM_WORLD);
  
  /* Loop waiting for incoming messages */
  long long sleep_nsecs = 1;
  while((nfinished < NTask-1) || (proc0_done==0))
    {
      /* If no local job is running, try to start one */
      pthread_mutex_lock( &job_running_mutex);
      int is_running = job_running;
      pthread_mutex_unlock( &job_running_mutex);
      if(is_running == 0)
        {
          /* Wait for thread which ran the previous job to finish */
          if(local_job >= 0)
            {
              pthread_join(job_thread, NULL);
              local_job = -1;
              /* Reset sleep timer if a local job finishes */
              sleep_nsecs = 1;
            }
          
          /* Report job completion (if any) */
          if(last_job>=0)printf("Job %d on process %d finished\n", last_job, 0);
          
          /* Check if we have jobs left to assign */
          if(next_to_assign <= ilast)
            {
              /* Launch the next job in a new thread */
              last_job = next_to_assign;
              printf("Running job %d on process %d\n", next_to_assign, 0);
              pthread_mutex_lock(&job_running_mutex);
              job_running = 1;
              pthread_mutex_unlock(&job_running_mutex);
              local_job = next_to_assign;
              pthread_create(&job_thread, NULL, 
                             &run_job, 
                             (void *) &local_job);
              next_to_assign += 1;
            }
          else
            {
              /* No more jobs left */
              proc0_done = 1;
              last_job = -1;
            }
        }
      
      /* 
         Check for job requests from other processes.
      */
      int num_complete, icomplete;
      MPI_Testsome(NTask-1, requests, &num_complete, index_complete, MPI_STATUSES_IGNORE);
      if((num_complete > 0) && (num_complete != MPI_UNDEFINED))
        {
          /* Loop over job requests which just arrived */
          for(icomplete=0;icomplete<num_complete;icomplete+=1)
            {
              /* Determine task we received from and index of any completed job */
              int source_task   = index_complete[icomplete]+1;
              int completed_job = recvbufs[index_complete[icomplete]];

              /* Report previous job completion (if any) */
              if(completed_job>=0)printf("Job %d on process %d finished\n", completed_job, source_task);
                  
              /* Check if we have jobs to hand out */
              int ijob;
              if(next_to_assign <= ilast)
                {
                  /* Choose job to assign */
                  printf("Running job %d on process %d\n", next_to_assign, source_task);
                  ijob = next_to_assign;
                  next_to_assign += 1;
                      
                  /* Post a new receive because we're expecting further messages from this task */
                  MPI_Irecv(&recvbufs[source_task-1], 1, MPI_INT, source_task, JOB_REQUEST_TAG,
                            MPI_COMM_WORLD, &requests[source_task-1]);
                }
              else
                {
                  /* No more jobs for this processor */
                  nfinished += 1;
                  ijob = -1;
                }
              /* Send the job index back */
              MPI_Send(&ijob, 1, MPI_INT, source_task, JOB_RESPONSE_TAG, MPI_COMM_WORLD);
              
              /* Reset the sleep delay to a small value */
              sleep_nsecs = 1;
            }
        }
      /* Go back to sleep */
      nanosec_sleep(sleep_nsecs);
      if(sleep_nsecs < MAX_SLEEP_NS)sleep_nsecs *= 2;
    }

  /* All request handles should be null by now */
  for(src=1;src<NTask;src+=1)
    {
      if(requests[src-1] != MPI_REQUEST_NULL)
        {
          printf("Something wrong here: all requests should be null at this point");
          terminate(1);
        }
    }
  free(requests);
  free(recvbufs);

#ifdef TERMINATE_SIGNAL
  /* 
     At this point, all jobs are complete so signal other tasks.
     This is just to avoid having them sit at 100% cpu in MPI_Barrier.
  */
  int i;
  int dummy = 1;
  for(i=1;i<NTask;i+=1)
    MPI_Send(&dummy, 1, MPI_INT, i, TERMINATION_TAG, MPI_COMM_WORLD);
#endif
}
