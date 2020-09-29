#ifndef RUNNER_H
#define RUNNER_H

/*
  Initialise the process we'll be using to run jobs

  Need to fork a new process and connect to its stdin and 
  stdout.

  Must be called before MPI_Init() in MPI programs, because
  fork() calls can interfere with some MPI implementations.
*/
void runner_init(void);


/*
  Terminate the job runner process
*/
void runner_shutdown(void);


/*
  Use the runner process to execute a command
  and return the exit code
*/
int runner_execute(char *command);

#endif

