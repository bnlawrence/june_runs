#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "job_info.h"
#include "command_file.h"

int read_command_file(char *fname) {
  
  /* Open the file */
  FILE *fd = fopen(fname,"r");
  if(!fd)return -1;

  /* Buffer to store a line */
  char line[COMMAND_LENGTH];
  
  /* Count non-empty lines in the file */
  int num_lines = 0;
  while(1) {

    /* Read a line and check for end of file */
    if(!fgets(line, COMMAND_LENGTH, fd))break;

    /* Abort if any line is too long */
    if(strlen(line)+1>=COMMAND_LENGTH) {
      fclose(fd);
      return -2;
    }
    
    num_lines += 1;
  }

  /* Allocate storage for all lines */
  command_line = malloc(sizeof(char *)*num_lines);
  for(int i=0; i<num_lines; i+=1) {
    command_line[i] = malloc(sizeof(char)*COMMAND_LENGTH);
  }

  /* Read the lines */
  num_lines = 0;
  fseek(fd, 0L, SEEK_SET);
  while(1) {

    /* Read a line and check for end of file */
    if(!fgets(line, COMMAND_LENGTH, fd))break;

    /* Abort if any line is too long */
    if(strlen(line)+1>=COMMAND_LENGTH) {
      fclose(fd);
      return -2;
    }
    
    /* Store the line, omitting any trailing newline */
    int ncopy = strnlen(line, COMMAND_LENGTH-1);
    if(line[ncopy-1]=='\n')ncopy -= 1;
    strncpy(command_line[num_lines], line, ncopy);
    command_line[num_lines][ncopy] = (char) 0;
    num_lines += 1;
  }

  fclose(fd);
  
  return num_lines;
}
