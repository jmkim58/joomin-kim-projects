#ifndef CISC5595_PROCESS_H
#define CISC5595_PROCESS_H

// process.h
//
// Class definition for a simulated process in our operating system
//

#include <iostream>
#include <string>

#include "nlohmann/json.hpp"

using namespace std;

class Process {
    public:

        // Constructor. The values default to an empty name and a process
        // ID of -1.
        Process(const string& processName = "", int priority = 100, int cpuNeeded = 0, int processId = -1 );

        // Setters and getters
        const string& processName() const;
        void setProcessName(const string& processName);

        const int processId() const;
        void setProcessId(int processId);

        const int priority() const;
        void setPriority(int priority);

        const int cpuNeeded() const;
        void setCpuNeeded(int cpuNeeded);

        // Note no setter for cpuUsed, because that is
        // internally managed
        const int cpuUsed() const;


        // run
        // Execute the given number of CPU cycles, if we need that many.
        // Return the number of cycles actually executed.
        int run(int numCycles);

        // This function will enable printing the Process using 
        // a construct like
        //    cout << process
        // The output should include all of the data members of the class
        // printed on a single line.
        friend std::ostream& operator<<(std::ostream& os, const Process& p);


    private:
        // Include private member data here

        int m_processId;
        std::string m_processName;
        int m_priority;
        int m_cpuNeeded;
        int m_cpuUsed;
};

// These methods will be used to populate a process object from
// json input files, or to output the json representation of a process
// object.
//
// See https://github.com/nlohmann/json/blob/develop/README.md#arbitrary-types-conversions
// for details on how to implement these functions

void to_json(nlohmann::json& j, const Process& p);
void from_json(const nlohmann::json& j, Process& p);

#endif // CISC5595_PROCESS_H