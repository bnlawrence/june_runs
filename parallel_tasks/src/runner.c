#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <unistd.h>
#include <sys/wait.h>

#include "runner.h"

static int pid;

#define PIPE_READ 0
#define PIPE_WRITE 1

/* Pipe to send input to child process */
static int input_pipe[2];

/* Pipe to receive output from child process*/
static int output_pipe[2];

/* Whether child process is running */
static int initialised = 0;

/*
  write() is not guaranteed to write all the data passed to it.
  This keeps calling write() until all bytes have been written.
*/
static ssize_t do_write(int filedes, const void *buf, size_t nbytes) {
  
  size_t nleft = nbytes;
  char *ptr = (char *) buf;

  while(nleft > 0) {
    ssize_t nwritten = write(filedes, ptr, nleft);
    if(nwritten < 0) {
      printf("write() call failed!\n");
      abort();
    } else {
      ptr += nwritten;
      nleft -= nwritten;
    }
  }
  
  return (ssize_t) nbytes;
}


/*
  read() is not guaranteed to read all the data passed to it.
  This keeps calling read() until all bytes have been read.
*/
static ssize_t do_read(int filedes, const void *buf, size_t nbytes) {
  
  size_t nleft = nbytes;
  char *ptr = (char *) buf;

  while(nleft > 0) {
    ssize_t nread = read(filedes, ptr, nleft);
    if(nread < 0) {
      printf("read() call failed!\n");
      abort();
    } else {
      ptr += nread;
      nleft -= nread;
    }
  }
  
  return (ssize_t) nbytes;
}


/*
  Receive and execute commands until told to stop
*/
static void runner_execute_commands(void) {

  while(1) {

    /* Receive command length */
    size_t length;
    do_read(input_pipe[PIPE_READ], &length, sizeof(size_t));

    /* Zero length means we should quit */
    if(length==0) {
      close(input_pipe[PIPE_READ]);
      close(output_pipe[PIPE_WRITE]);
      exit(0);
    }

    /* Read the command */
    char *str = malloc(length);
    do_read(input_pipe[PIPE_READ], str, length);
    if(str[length-1] != 0) {
      printf("Command is not null terminated!\n");
      abort();
    }

    /* Run the command */
    int return_code = system(str);
    free(str);

    /* Send the return code back */
    do_write(output_pipe[PIPE_WRITE], &return_code, sizeof(int));

  }  
}


/*
  Terminate the job runner process
*/
void runner_shutdown(void) {
  
  if(!initialised)return;

  /* Send shutdown signal (zero command length) */
  size_t length = 0;
  do_write(input_pipe[PIPE_WRITE], &length, sizeof(size_t));

  /* Wait for child process to complete and check return code */
  int status;
  int failed = 0;
  waitpid(pid, &status, 0);
  if(WIFEXITED(status)) {
    if(WEXITSTATUS(status)!=0) failed = 1;
  } else {
    failed = 1;
  }
  if(failed) {
    printf("Subprocess had non-zero exit code!\n");
    abort();
  }

  /* Close stdin/stdout file descriptors */
  close(input_pipe[PIPE_WRITE]);
  close(output_pipe[PIPE_READ]);

  initialised = 0;

  return;
}


/*
  Initialise the process we'll be using to run jobs

  Need to fork a new process and connect to its stdin and 
  stdout.

  Must be called before MPI_Init() in MPI programs, because
  fork() calls can interfere with some MPI implementations.
*/
void runner_init(void) {
  
  /* 
     Create the pipes to communicate with the child process:
  */
  if (pipe(input_pipe) == -1) {
    printf("Failed creating pipe\n");
    abort();
  }
  if (pipe(output_pipe) == -1) {
    printf("Failed creating pipe\n");
    abort();
  }
    
  /* Fork the new process */
  pid = fork();
  if (pid == -1) {

    /* Only get here if fork() failed */
    printf("Fork call failed!\n");
    abort();

  } else if (pid == 0) {

    /* Child will not write to input pipe or read from output pipe */
    close(input_pipe[PIPE_WRITE]);
    close(output_pipe[PIPE_READ]);

    /* Wait for and execute any commands */
    runner_execute_commands();

    /* Terminate */
    exit(0);
    
  } else {
    
    /* Kill child process if we exit */
    atexit(runner_shutdown);

    /* Parent will not read child's input or write to its output */
    close(input_pipe[PIPE_READ]);
    close(output_pipe[PIPE_WRITE]);
    
  }
  
  initialised = 1;

  return;
}


/*
  Use the runner process to execute a command
  and return the exit code
*/
int runner_execute(char *command) {
  
  /* Send the command length, including null terminator */
  size_t length = strlen(command) + 1;
  do_write(input_pipe[PIPE_WRITE], &length, sizeof(ssize_t));

  /* Send the command */
  do_write(input_pipe[PIPE_WRITE], command, length);
  
  /* Receive the return code */
  int retcode;
  do_read(output_pipe[PIPE_READ], &retcode, sizeof(int));
  
  return retcode;
}


