#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#include "runner.h"

void terminate(int err) {
  
  runner_shutdown();
  fflush(stdout);
  fflush(stderr);

  MPI_Abort(MPI_COMM_WORLD, err);
  
}
