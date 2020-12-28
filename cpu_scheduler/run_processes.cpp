// Main program for assignment 3 - Process Scheduling
//
// This program will be run as follows:
//
// run_processes --algorithm=<algo> --num_threads=<threads> <json_files>
//    Where <algo> will be either "fcfs" or "priority",
//          <threads> is the number of threads to run, and 
//          <json_files> is a set of one or more processes to run

#include "cpu_scheduler_base.h"
#include "cpu_scheduler_fcfs.h"
#include "cpu_scheduler_priority_rr.h"
#include "process.h"
#include "process_table.h"

#include <getopt.h>
#include <iostream>
#include <fstream>

using namespace std;

// Thread functions take a single pointer to any data that
// they might need to use. 
// Add members to this structure to represent all the data your
// thread function will need.
struct processRunnerData {
	CpuSchedulerBase * scheduler;
	ProcessTable *table;
   
};


// This is the function that will be executed by your threads.

void * processRunner(void * data)
{

	
	
    // Here you take the pointer to your thread data and convert it
    // to the struct defined above.
    struct processRunnerData * d = (struct processRunnerData *) data;

    // Implement your scheduling loop in this function.
    // The thread will have the same logic you used in the 
    // run_processes program in assignment 3.
    //
    // NOTE: Remember the way the nextProcess() method has changes
    //       in this assignment. Make sure you make the appropriate changes 
    //       here to reflect the changes to nextProcess() and the new
    //       requeueProcess() method.

    int processToRun;
    while((processToRun = d->scheduler->nextProcess()) != -1) {
        Process &p = d->table->find(processToRun);
        if (p.processId() == -1) {
            cerr << "Error finding process " << processToRun << " to run it" << endl;
            pthread_exit(NULL);
        }

        p.run(1);
    		
        cout << p.processName() << "(" << p.priority() << "): " << p.cpuUsed() << "/" << p.cpuNeeded() << endl;
        if (p.cpuUsed() != p.cpuNeeded()) {
            d->scheduler->requeueProcess(p);
        }
    }

    


    // Once you have completed the implementation of your 
    // scheduling, this causes your thread to exit.
    pthread_exit(NULL);
}


int main(int argc, char ** argv)
{
    ProcessTable table;             // Will contain the processes to be scheduled
    CpuSchedulerBase * scheduler;   // Scheduler that will be used 
    string algorithm;               // String choosing which scheduling algorithm to use
    int numThreads = 10;             // Number of threads to start

    // Copy your command line processing from assignment 3, and update it to
    // add the new argument (--num_threads) for assignment 4.

    // These external variables are defined and used by the getopt()
    // functions.
    extern char *optarg;
    extern int optind, opterr, optopt;

    // This identifies what the accepted arguments are; in this case,
    // we accept a single required argument "algorithm".
    // 
    // Note that the last element of the structure (with a value of 1 below)
    // is the value that will be returned by getopt_long when it recognizes
    // the argument.
    static struct option program_options[] = {
        {"algorithm", required_argument, 0, 1},
        {0, 0, 0, 0}
    };

    int option_index = 0;

    int c = 0;
    
    while(c != -1) {
        // Returns the 
        c = getopt_long(argc, argv, "",
                        program_options, &option_index);

        switch (c) {
            case -1:
                // getopt_long returns -1 when it reaches the end of
                // the argument list.
                break;
            case 1:
                // algorithm
                algorithm = optarg;
                if ((algorithm != "fcfs") && (algorithm != "priority")) {
                    cerr << "Error: Invalid value '" << algorithm << "' for --algorithm" << endl;
                    cerr << "       Legal values are 'fcfs' and 'priority'" << endl;
                    return(1);
                }
                break;
            case '?':
                // Unknown argument
                cerr << "Error: Invalid argument: " << argv[optind] << endl;
                return(3);          
        }     
    }



    // Allocate a scheduling object based on the
    // chosen algorithm.
    if (algorithm == "fcfs") {
        scheduler = (CpuSchedulerBase *) new CpuSchedulerFcfs;
    } else if (algorithm == "priority") {
        scheduler = (CpuSchedulerBase *) new CpuSchedulerPriorityRr;
    } else {
        cerr << "No algorithm specified!" << endl;
        return(9);
    }

    // The rest of the arguments will be files in json format containing 
    // specifications for processes to be added to the table.

    // Copy your code for loading the processes from the input json files and 
    // adding them to your process table and scheduler, using the same
    // logic as assignment 3.
 	for (; optind < argc; ++optind) {
        
        // Open the input file. 
        ifstream in(argv[optind]);
        if (! in.good()) {
            cerr << "Failed to open file " << argv[optind] << endl;
            return(4);
        }

        nlohmann::json process_json;

        // Read the file into the json object.
        // If the input file is not in json format, this will
        // throw an exception

        try {
            in >> process_json;
        } catch (nlohmann::json::exception& e) {
            cerr << "Error: " << e.what() << endl
                 << "exception id: " << e.id << endl;
                return(5);
        }


        // Iterate the entries in the input json,
        // converting each one to a Process object
        for(auto proc = process_json["processes"].begin(); 
            proc != process_json["processes"].end();
            ++proc) {
            
            Process p;

            // Note that if there is an extra unrecogized keyword
            // in the input json, it will simply be ignored, and no
            // exception will be thrown.
            try {
                proc->get_to(p);
            } catch (nlohmann::json::exception& e) {
                // output exception information
                cerr << "Error: " << e.what() << endl
                     << "exception id: " << e.id << std::endl;
                return(6);
            } catch (std::invalid_argument& e) {
                cerr << "Error: " << e.what() << endl;
                return(7);
            }

            if (!table.add(p)) {
                cerr << "Failed to add process: " << p.processName() << endl;
                return(8);
            }
        }        
    }


    // Iterate the set of processes in the ProcessTable and add
    // each process to the scheduler. Processes should be added 
    // in order of process_id, from 1 to the maximum table size

    for (int i = 1; i < table.getSize() + 1; ++i) {
        Process p = table.find(i);
        if (p.processId() != -1) {
            scheduler->addProcess(p);
        }
    }



    // Now start the threads that will perform the actual execution
    // of the processes

    pthread_t threads[numThreads];

    for (int i = 0; i < numThreads; ++i) {
        // Set up the data that will be passed to each thread
        struct processRunnerData * d = new struct processRunnerData;
        // Populate the thread data object with the data that 
        // the thread will need to execute.
        d->scheduler = scheduler;
        d->table = &table;
	
        // This creates the thread
        pthread_create(&threads[i], NULL, processRunner, (void *) d);
    }

    // Now wait for the threads to finish
    void * ret;
    for (int i = 0; i < numThreads; ++i) {
        pthread_join(threads[i], &ret);
    }
   
    // Return 0 when complete
    return 0;
}
