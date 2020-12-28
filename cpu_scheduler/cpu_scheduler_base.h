#ifndef CISC5595_CPU_SCHEDULER_BASE_H
#define CISC5595_CPU_SCHEDULER_BASE_H

// cpu_scheduler_base.h
//
// Abstract base class of a CPU scheduler.

#include "process.h"

class CpuSchedulerBase {

    public:

        // Add a new process to the set of processes to be
        // scheduled.
        virtual bool addProcess(Process process) = 0;

        // Put a process that was previously scheduled back
        // into the scheduler. 
        virtual bool requeueProcess(Process process) = 0;

        // Get the process id of the next process to be scheduled
        // If no processes remain to schedule, return -1.
        // NOTE: The process being scheduled should be removed
        //       as part of this function
        virtual int nextProcess() = 0;
};

#endif