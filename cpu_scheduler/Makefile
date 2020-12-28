CPPFLAGS = -g -I.
LDFLAGS = -lpthread

libsrc = process.cpp process_table.cpp cpu_scheduler_fcfs.cpp cpu_scheduler_priority_rr.cpp
libobj = $(libsrc:.cpp=.o)

all: run_processes $(libobj)

run_processes: run_processes.o $(libobj)
	$(CXX) -o $@ run_processes.o $(libobj) $(LDFLAGS)

.PHONY: clean
clean:
	rm -f process_test process_test.o run_processes run_processes.o $(libobj)