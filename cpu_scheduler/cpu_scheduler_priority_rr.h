#ifndef CISC5595_CPU_SCHEDULER_PRIORITY_H
#define CISC5595_CPU_SCHEDULER_PRIORITY_H

// cpu_scheduler_fcfs.h
//
// First-come first-served CPU scheduler.

#include "cpu_scheduler_base.h"
#include "process.h"

#include <map>
#include <deque>
#include <mutex>

using namespace std;

class CpuSchedulerPriorityRr : public CpuSchedulerBase {

    public:

        // Add a new process to the set of processes to be
        // scheduled.
        virtual bool addProcess(Process process);

        // Put a process that was previously scheduled back
        // into the scheduler. 
        virtual bool requeueProcess(Process process);

        // Get the process id of the next process to be scheduled
        // If no processes remain to schedule, return -1.
        // NOTE: The process being scheduled should be removed
        //       as part of this function
        virtual int nextProcess();

    private:
        // Use the same data structure as you used for this class
        // in assignment 3 (or copy from the solution for assignment 3
        // once that is available)
	map<int, deque<int> > m_priorityMap;
	
        // This mutex should be used to synchronize access to 
        // class methods
        mutex m_mutex;
};

#endif
