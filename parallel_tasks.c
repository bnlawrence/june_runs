#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <mpi.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>

#define TERMINATE_SIGNAL

/* Number of processors, ID of this processor */
static int NTask;
static int ThisTask;

/* Flag to signal job completion */
pthread_mutex_t job_running_mutex = PTHREAD_MUTEX_INITIALIZER;
int job_running = 0;

/* Maximum length of command to execute */
#define COMMAND_LENGTH 4096
static char command[COMMAND_LENGTH];

/* MPI tags */
#define JOB_REQUEST_TAG  1
#define JOB_RESPONSE_TAG 2
#define TERMINATION_TAG  3

/* Range of job indexes to do */
static int ifirst, ilast;

/* Table of job results */
int *job_result;

/* Job elapsed times */
double *job_time;

/* Maximum time to wait when polling for incoming messages (nanosecs) */
#define MAX_SLEEP_NS 1000000000

/* Sleep for specified number of nanoseconds */
void nanosec_sleep(long long n)
{
  struct timespec ts;
  ts.tv_sec  = n / 1000000000;
  ts.tv_nsec = n % 1000000000;
  nanosleep(&ts, NULL);
}

/* Get wall clock time in seconds as a double */
double get_time()
{
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return ((double) t.tv_nsec)/1.0e9+t.tv_sec;
}

/*
  Check to see what format specifier a string contains,
  if any. Returns index of the character at the end of the
  format string or 0 if there isn't one.
*/
size_t identify_format(char *str, size_t len)
{
  size_t i = 0;

  /* Advance to first '%' sign */
  while((i < len-1) && (str[i] != '%'))
    i += 1;
  
  /* Return 0 if we didn't find a % or we're at the end */
  if((str[i] != '%') || (i >= len-1))
    return 0;
  
  /* Advance to type character/flags etc */
  i += 1;
  if(i >= len)return 0;

  /* Skip over any flags */
  while((i < len-1) && 
	((str[i]=='+') ||
	 (str[i]=='-') ||
	 (str[i]==' ') ||
	 (str[i]=='#') ||
	 (str[i]=='0')))
    i += 1;
  
  /* Skip width */
  while((i < len-1) && 
	(str[i] >= '0') && 
	(str[i] <= '9'))
    i += 1;

  /* Skip precision */
  if(str[i] == '.')
    {
      /* Skip over dot */
      i+= 1;
      if(i>=len-1)
	return 0;
      
      /* Skip any digits */
      while((i < len-1) && 
	    (str[i] >= '0') && 
	    (str[i] <= '9'))
	i += 1;
    }

  /* Skip modifier */
  while((i < len-1) && 
	((str[i]=='h') ||
	 (str[i]=='l') ||
	 (str[i]=='L')))
    i += 1;

  /* Should now have type character */
  switch (str[i])
    {
    case 'd':
    case 'i':
    case 'f':
    case 'e':
    case 'E':
    case 'g':
    case 'G':
      return i;
    break;
    default:
      return 0;
      break;
    }
}



void *run_job(void *ptr)
{
  char cmd_exec[COMMAND_LENGTH];
  char tmp[COMMAND_LENGTH];
  int ijob = *((int *) ptr);
  size_t exec_offset = 0;
  size_t len = strlen(command);
  size_t offset = 0;
  size_t fpos;
  int return_code;

  /* 
     Construct command by substituting in job index
     wherever we find an int or double format specifier
  */
  while(offset < len)
    {
      fpos = identify_format(command+offset, len-offset);
      if(fpos == 0)
	{
	  /* No format strings left, so just copy */
	  strncpy(cmd_exec+exec_offset, command+offset, 
		 COMMAND_LENGTH-exec_offset);
	  offset = len;
	}
      else
	{
	  /* Have to sub in the job number */
	  strncpy(tmp, command+offset, fpos+1);
	  tmp[fpos+1] = (char) 0;

	  switch (command[offset+fpos])
	    {
	    case 'd':
	    case 'i':
	      {
		sprintf(cmd_exec+exec_offset, tmp, ijob);
		exec_offset = strlen(cmd_exec);
		offset += fpos + 1;
	      }
	    break;
	    case 'f':
	    case 'e':
	    case 'E':
	    case 'g':
	    case 'G':
	      {
		sprintf(cmd_exec+exec_offset, tmp, (double) ijob);
		exec_offset = strlen(cmd_exec);
		offset += fpos + 1;
	      }
	    break;
	    default:
	      /* Can't handle this format, so just copy */
	      strncpy(cmd_exec+exec_offset, command+offset, 
		     COMMAND_LENGTH-exec_offset);
	      offset = len;
	    }
	}
    }
  
  /* Run the command */
  double start_time = get_time();
  return_code = system(cmd_exec);
  double finish_time = get_time();

  /* On certain return codes, we should abort all jobs */
  switch (return_code)
    {
    case 9:
    case 137:
      fprintf(stderr, "Job %d returned code %d (assumed to mean SIGKILL). Aborting all jobs.\n", ijob, return_code);
      MPI_Abort(MPI_COMM_WORLD, 1);
      break;
    }

  /* Return job result and run time */
  job_result[ijob-ifirst] = return_code;
  job_time[ijob-ifirst] = (finish_time-start_time);
  pthread_mutex_lock( &job_running_mutex);
  job_running = 0;
  pthread_mutex_unlock( &job_running_mutex);
  return NULL;
}



int main(int argc, char *argv[])
{
  int njobs_tot;
  int next_to_assign;
  int ijob;
  int *job_result_all;
  double *job_time_all;
  int nfailed;
  int provided;
  int last_job = -1;

  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  if(provided<MPI_THREAD_FUNNELED)
    {
      printf("This program needs MPI with at least MPI_THREAD_FUNNELED thread support\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
  MPI_Comm_size(MPI_COMM_WORLD, &NTask);

  /* Read command line args */
  if(ThisTask==0)
    {
      if(argc != 4)
	{
	  printf("Usage: parallel_tasks ifirst ilast command\n");
	  MPI_Abort(MPI_COMM_WORLD, 1);
	}
      sscanf(argv[1], "%d", &ifirst);
      sscanf(argv[2], "%d", &ilast);
      strncpy(command, argv[3], COMMAND_LENGTH);
      command[COMMAND_LENGTH-1] = (char) 0;

      if((ifirst < 0) || (ilast < 0))
	{
	  printf("Job index must be non-negative!\n");
	  MPI_Abort(MPI_COMM_WORLD, 1);  
	}
    }

  if(ThisTask==0)
    printf("Parallel tasks - command is: %s\n", command);

  /* Broadcast arguments */
  MPI_Bcast(&ifirst, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&ilast,  1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(command, COMMAND_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);

  /* Initial number of jobs to assign */
  njobs_tot      = ilast-ifirst+1;
  next_to_assign = ifirst;

  /* Initialize results table */
  job_result = malloc(sizeof(int)*njobs_tot);
  job_time   = malloc(sizeof(double)*njobs_tot);
  for(ijob=0;ijob<njobs_tot;ijob+=1)
    {
      job_result[ijob] = INT_MIN;
      job_time[ijob] = -1.0;
    }

  /* Start the clock */
  double start_time = get_time();

  if(ThisTask==0)
    {
      /*
	Processor 0 needs to handle job requests and 
	run its own jobs at the same time. Jobs on this
	process are run in a separate thread so we can continue
	responding to job requests. The main thread spends most
	of its time sleeping to avoid taking cpu time from the jobs.
      */

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
              MPI_Abort(MPI_COMM_WORLD, 1);
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
  else
    {
      /*
	Processors other than 0 request and execute jobs
	until there are none left (signalled by -ve job ID).

        To request a job we send the index of the last job we
        completed (or -1 if there was no job) to rank 0.
      */
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

  /* If not using a terminate signal, all wait here once their jobs are done */
  MPI_Barrier(MPI_COMM_WORLD);

  /* Stop the clock */
  double finish_time = get_time();

  /* Gather exit codes and run times */
  job_result_all = malloc(sizeof(int)*njobs_tot);
  MPI_Reduce(job_result, job_result_all, njobs_tot, MPI_INT, MPI_MAX, 0, 
             MPI_COMM_WORLD);
  job_time_all = malloc(sizeof(double)*njobs_tot);
  MPI_Reduce(job_time, job_time_all, njobs_tot, MPI_DOUBLE, MPI_MAX, 0, 
             MPI_COMM_WORLD);
  
  if(ThisTask==0)
    {
      /* Calculate idle time */
      double total_time = (finish_time-start_time)*NTask;
      double idle_time = total_time;
      double max_time = 0.0;
      double min_time = total_time;
      for(ijob=0;ijob<njobs_tot;ijob+=1)
        {
          idle_time -= job_time_all[ijob];
          min_time = job_time_all[ijob] < min_time ? job_time_all[ijob] : min_time;
          max_time = job_time_all[ijob] > max_time ? job_time_all[ijob] : max_time;
        }

      /* Generate a report */      
      printf("\nElapsed time:\n");
      printf("-------------\n");
      printf("\nRan jobs %d to %d in %f seconds (total no. of jobs = %d)\n", 
             ifirst, ilast, (finish_time-start_time), njobs_tot);
      printf("Shortest job: %f seconds\n", min_time);
      printf("Longest  job: %f seconds\n", max_time);
      printf("Idle time: %.2f%%\n", idle_time/total_time*100);
      printf("\nJob exit codes:\n");
      printf("---------------\n\n");
      nfailed = 0;
      for(ijob=0;ijob<njobs_tot;ijob+=1)
        {
          if(job_result_all[ijob] != 0)
            {
              nfailed += 1;
              printf("ERROR: Job %d returned non-zero exit code %d\n", ijob+ifirst, job_result_all[ijob]);
            }
        }
      if(nfailed>0)printf("\n");
    }
  free(job_result_all);
  free(job_result);
  free(job_time_all);
  free(job_time);
  MPI_Bcast(&nfailed, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if(ThisTask==0)
    {
      if(nfailed==0)
        printf("All jobs completed with zero exit code\n");
      else
        printf("Number of jobs with non-zero exit code: %d of %d\n", nfailed, njobs_tot);
    }
  
  MPI_Finalize();
  
  /* All MPI tasks return non-zero exit code if any job failed */
  if(nfailed==0)
      return 0;
  else
      return 1;
}
