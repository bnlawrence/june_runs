add_executable(parallel_tasks 
  main.c identify_format.c job_info.c run_job.c get_time.c nanosec_sleep.c
  command_file.c master_task.c worker_task.c runner.c terminate.c )

target_link_libraries(parallel_tasks PRIVATE MPI::MPI_C)

add_executable(testprog testprog.c runner.c)
