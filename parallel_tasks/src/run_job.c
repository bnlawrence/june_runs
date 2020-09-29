#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <mpi.h>

#include "run_job.h"
#include "job_info.h"
#include "identify_format.h"
#include "get_time.h"
#include "runner.h"
#include "terminate.h"

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

     TODO: change sprintfs to snprintfs
  */
  while(offset < len)
    {
      fpos = identify_format(command+offset, len-offset);
      if(fpos == 0)
	{
	  /* No format strings left, so just copy */
	  strncpy(cmd_exec+exec_offset, command+offset, 
		 COMMAND_LENGTH-exec_offset);
          cmd_exec[COMMAND_LENGTH-1] = (char) 0;
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
                /* This is an integer format specifier */
                if(have_command_file)
                  {
                    printf("Found integer specifier when expecting a string\n");
                    terminate(1);
                  } 
                else 
                  {
                    snprintf(cmd_exec+exec_offset, COMMAND_LENGTH-exec_offset,
                             tmp, ijob);
                    exec_offset = strlen(cmd_exec);
                    offset += fpos + 1;
                  }
              }
              break;
            case 'f':
	    case 'e':
	    case 'E':
	    case 'g':
	    case 'G':
	      {
                /* This is a float format specifier */
                if(have_command_file)
                  {
                    printf("Found float specifier when expecting a string\n");
                    terminate(1);
                  } 
                else 
                  {
                    snprintf(cmd_exec+exec_offset, COMMAND_LENGTH-exec_offset,
                             tmp, (double) ijob);
                    exec_offset = strlen(cmd_exec);
                    offset += fpos + 1;
                  }
	      }
              break;
            case 's':
              {
                /* This is a string format specifier */
                if(!have_command_file)
                  {
                    printf("Found string specifier when expecting int/float\n");
                    terminate(1);
                  } 
                else
                  {
                    snprintf(cmd_exec+exec_offset, COMMAND_LENGTH-exec_offset,
                             tmp, command_line[ijob]);
                    exec_offset = strlen(cmd_exec);
                    offset += fpos + 1;
                  }
              }
              break;
            default:
              {
                /* Can't handle this format, so just copy */
                strncpy(cmd_exec+exec_offset, command+offset, 
                        COMMAND_LENGTH-exec_offset);
                cmd_exec[COMMAND_LENGTH-1] = (char) 0;
                offset = len;
              }
            }
	}
    }
  
  /* Run the command */
  double start_time = get_time();
  return_code = runner_execute(cmd_exec);
  double finish_time = get_time();

  /* On certain return codes, we should abort all jobs */
  switch (return_code)
    {
    case 9:
    case 137:
      fprintf(stderr, "Job %d returned code %d (assumed to mean SIGKILL). Aborting all jobs.\n", ijob, return_code);
      terminate(1);
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
