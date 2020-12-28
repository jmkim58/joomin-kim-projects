#include "cpu_scheduler_fcfs.h"
#include "process.h"

// Add a process to be scheduled.
// Since this is first come, first served, this just adds
// the process ID of the process to be scheduled to 
// the queue.
bool CpuSchedulerFcfs::addProcess(Process process)
{
	std::lock_guard<std::mutex> guard(this->m_mutex);
	m_queue.push_back(process.processId());
    	return true;
}

// Put a process that was previously scheduled back
// into the scheduler. Since this is first come, first-served,
// we want to requeue the process at the beginning of the
// list rather than the end (as we do with add)
bool CpuSchedulerFcfs::requeueProcess(Process process)
{
	std::lock_guard<std::mutex> guard(this->m_mutex);
	m_queue.push_front(process.processId());
}


// Find the next process to run
int CpuSchedulerFcfs::nextProcess()
{
	
	std::lock_guard<std::mutex> guard(this->m_mutex);
	if (! m_queue.empty()) {
		int id = m_queue.front();
		m_queue.pop_front();
		return id;
	    } 
	else {
		return -1;
	}

}
