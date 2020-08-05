CC      = mpicc
CFLAGS  = -g -O0 -Wall -pthread

.SUFFIXES:
.SUFFIXES: .c .o

.c.o:
	$(CC)  $(CFLAGS)   -c $< -o $*.o

parallel_tasks:	parallel_tasks.o
	$(CC) $(CFLAGS) parallel_tasks.o -o parallel_tasks $(LDFLAGS)

clean:
	\rm -f *.o parallel_tasks
