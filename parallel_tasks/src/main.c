#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <mpi.h>
#include <unistd.h>
#include <pthread.h>

#include "identify_format.h"
#include "job_info.h"
#include "run_job.h"
#include "get_time.h"
#include "nanosec_sleep.h"
#include "master_task.h"
#include "worker_task.h"
#include "command_file.h"
#include "runner.h"
#include "terminate.h"


void usage(void) {
  printf("\n");
  printf("  parallel_tasks usage:\n");
  printf("\n");
  printf("    mpirun -np N parallel_tasks ifirst ilast command_template\n");
  printf("\n");
  printf("    This runs the supplied command once for each index in the range ifirst\n");
  printf("    to ilast, substituting the index in place of any C integer format\n");
  printf("    specifiers (e.g. %%d).\n");
  printf("\n");
  printf("  OR\n");
  printf("\n");
  printf("    mpirun -np N parallel_tasks command_file command_template\n");
  printf("\n");
  printf("    For each line in the file command_file, this substitutes the line\n");
  printf("    into command_template in place of any string format specifiers (e.g. %%s)\n");
  printf("    and runs the command.\n");
  printf("\n");
  printf("  In both cases the command should be quoted to prevent the shell splitting\n");
  printf("  it into multiple arguments and up to N commands are run simultaneously\n");
  printf("\n");
}


int main(int argc, char *argv[])
{
  int njobs_tot;
  int ijob;
  int *job_result_all;
  double *job_time_all;
  int nfailed;

  /* Fork job runner process (it's not safe to use system() after MPI_Init) */
  runner_init();

  /* Initialize MPI with the thread support we need */
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  if(provided<MPI_THREAD_FUNNELED)
    {
      printf("This program needs MPI with at least MPI_THREAD_FUNNELED thread support\n");
      terminate(1);
    }
  MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
  MPI_Comm_size(MPI_COMM_WORLD, &NTask);

  /* Read command line args */
  int ok = 1;
  if(ThisTask==0)
    {
      switch(argc)
        {
        case 3:
          /* We have command_file, command_template */
          ifirst = 0;
          ilast = read_command_file(argv[1])-1;
          if(ilast < 0) {
            printf("Failed to read any lines from %s\n", argv[1]);
            terminate(1);
          }
          strncpy(command, argv[2], COMMAND_LENGTH);
          command[COMMAND_LENGTH-1] = (char) 0;
          have_command_file = 1;
          break;
        case 4:
          /* We have ifirst, ilast, command_template */
          sscanf(argv[1], "%d", &ifirst);
          sscanf(argv[2], "%d", &ilast);
          strncpy(command, argv[3], COMMAND_LENGTH);
          command[COMMAND_LENGTH-1] = (char) 0;
          if((ifirst < 0) || (ilast < 0))
            {
              printf("Job index must be non-negative!\n");
              terminate(1);
            }
          have_command_file = 0;          
          break;
        default:
          /* Invalid number of arguments */
	  usage();
          ok = 0;
        }
    }
  MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if(!ok) {
    MPI_Finalize();
    exit(0);
  }

  if(ThisTask==0)
    printf("Parallel tasks - command is: %s\n", command);

  /* Broadcast arguments */
  MPI_Bcast(&ifirst, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&ilast,  1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(command, COMMAND_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);
  MPI_Bcast(&have_command_file, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if(have_command_file)
    {
      if(ThisTask!=0)command_line = malloc(sizeof(char *)*(ilast-ifirst+1));
      for(int i=ifirst; i<=ilast; i+=1)
        {
          if(ThisTask!=0)command_line[i] = malloc(sizeof(char)*COMMAND_LENGTH);
          MPI_Bcast(command_line[i], COMMAND_LENGTH, MPI_CHAR, 0, MPI_COMM_WORLD);      
        }
    }

  /* Initial number of jobs to assign */
  njobs_tot = ilast-ifirst+1;

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

  /* Run jobs until all are done */
  if(ThisTask==0)
    {
      master_task();
    }
  else
    {
      worker_task();
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
              printf("ERROR: Job %d returned non-zero exit code %d\n",
                     ijob+ifirst, job_result_all[ijob]);
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

  /* Terminate job runner process */
  runner_shutdown();
  
  /* All MPI tasks return non-zero exit code if any job failed */
  if(nfailed==0)
      return 0;
  else
      return 1;
}
