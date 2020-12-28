// process.cpp
//
// Include your implementation of the Process class here.

#include <string>
#include <iostream>
#include <sstream>

#include "process.h"
#include "nlohmann/json.hpp"

using namespace std;

Process::Process(const string& processName, int priority, int cpuNeeded, int processId) :
    m_processName(processName), 
    m_priority(priority), 
    m_cpuNeeded(cpuNeeded), 
    m_processId(processId), 
    m_cpuUsed(0)
{
    // Nothing to do here
}

const string& Process::processName() const
{
    return m_processName;
}

void Process::setProcessName(const string& processName)
{
    m_processName = processName;
}

const int Process::processId() const
{
    return m_processId;
}

void Process::setProcessId(int processId)
{
    m_processId = processId;
}


const int Process::priority() const
{
    return m_priority;
}

void Process::setPriority(int priority)
{
    m_priority = priority;
}


const int Process::cpuNeeded() const
{
    return m_cpuNeeded;
}


void Process::setCpuNeeded(int cpuNeeded)
{
    m_cpuNeeded = cpuNeeded;
}

const int Process::cpuUsed() const
{
    return m_cpuUsed;
}


int Process::run(int numCycles)
{
    int toRun;

    if ((m_cpuNeeded - m_cpuUsed) >= numCycles) {
        toRun = numCycles;
    } else {
        toRun = m_cpuNeeded - m_cpuUsed;
    }

    m_cpuUsed += toRun;
    return toRun;
}



// Friend function to print the object
std::ostream& operator<<(std::ostream& os, const Process& p)
{
    nlohmann::json j;
    to_json(j, p);
    os << j;
    return os;
}

// Functions to convert a process to and from json
void to_json(nlohmann::json& j, const Process& p)
{
    j["process_name"] = p.processName();
    j["process_id"] = p.processId();
    j["priority"] = p.priority();
    j["cpu_needed"] = p.cpuNeeded();
    j["cpu_used"] = p.cpuUsed();
}

void from_json(const nlohmann::json& j, Process& p)
{
    if (j.contains("process_name")) {
        p.setProcessName(j["process_name"]);
    } else {
        ostringstream o;
        o << "Missing process_name in json input: " << j;
        throw(std::invalid_argument(o.str()));
    }

    if (j.contains("process_id")) {
        p.setProcessId(j["process_id"]);
    }

    if (j.contains("priority")) {
        p.setPriority(j["priority"]);
    }

    if (j.contains("cpu_needed")) {
        p.setCpuNeeded(j["cpu_needed"]);
    }
}