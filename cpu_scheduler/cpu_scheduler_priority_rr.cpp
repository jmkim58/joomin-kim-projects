#include "cpu_scheduler_priority_rr.h"
#include "process.h"

#include <map>
#include <deque>

// Add a process to be scheduled.
// Since this is first come, first served, this just adds
// the process ID of the process to be scheduled to 
// the queue.

bool CpuSchedulerPriorityRr::addProcess(Process process)
{
	std::lock_guard<std::mutex> guard(this->m_mutex);
	m_priorityMap[process.priority()].push_back(process.processId());
    	return true;
}

// Put a process that was previously scheduled back
// into the scheduler. 
bool CpuSchedulerPriorityRr::requeueProcess(Process process)
{
	std::lock_guard<std::mutex> guard(this->m_mutex);
	m_priorityMap[process.priority()].push_back(process.processId());
    	return true;
}


// Find the next process to run
int CpuSchedulerPriorityRr::nextProcess()
{
    std::lock_guard<std::mutex> guard(this->m_mutex);
    // Find the first priority bucket with entries
    auto iter = m_priorityMap.begin();
    while (iter != m_priorityMap.end()) {
        if (! iter->second.empty()) {
            int result = iter->second.front();

            // Now pop it off the front and push it to back
            iter->second.pop_front();
            return result;
        }
        ++iter;
    }

    return -1;
 
}
