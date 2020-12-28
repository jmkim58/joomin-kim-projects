// load_table.cpp
// Program to load the process table from one or more input files
//
// Arguments:
//    --table_size <size> 

#include <getopt.h>
#include <string>
#include <fstream>
#include <iostream>

#include "nlohmann/json.hpp"
#include "process_table.h"
#include "process.h"

using namespace std;


int
main(int argc, char ** argv)
{

    // Table size defaults to 100 if not specified on the command line
    int tableSize = 100;

    //
    // Process command line arguments
    //
   
    // These external variables are defined and used by the getopt()
    // functions.
    extern char *optarg;
    extern int optind, opterr, optopt;

    // This identifies what the accepted arguments are; in this case,
    // we accept a single optional argument "table_size".
    // 
    // Note that the last element of the structure (with a value of 1 below)
    // is the value that will be returned by getopt_long when it recognizes
    // the table_size argument.
    static struct option program_options[] = {
        {"table_size", optional_argument, 0, 1},
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
                // table_size
                tableSize = atoi(optarg);
                if (tableSize <= 0) {
                    cerr << "Error: --table_size must be a positive number!" << endl;
                    return(1);
                }
                break;
            case '?':
                // Unknown argument
                cerr << "Error: Invalid argument: " << argv[optind] << endl;
                return(3);          
        }     
    }


    // Initialize a ProcessTable with the table size obtained via
    // the command-line arguments
    ProcessTable table(tableSize);

    // The rest of the arguments will be files in json format containing 
    // specifications for processes to be added to the table.

    // After the getopt_long processing, optind will be the index of the
    // next argument to read.

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

    // Print the contents of the table to cout
    cout << table << endl;

    return 0;
}