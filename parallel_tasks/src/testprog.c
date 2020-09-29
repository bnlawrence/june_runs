#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include "runner.h"


int main(int argc, char *argv[]) {
  
  runner_init();

  runner_execute("echo hello 1 && sleep 10");
  runner_execute("echo hello 2");
  runner_execute("echo hello 3");
  runner_execute("echo hello 4");

  runner_shutdown();
  
  return 0;
}
