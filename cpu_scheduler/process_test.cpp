// process_test.cpp
//
// Test program for the Process and ProcessTable classes

#include <process.h>
#include <process_table.h>

#include <iostream>
#include <sstream>

using namespace std;

//
// Create a series of tests for your Process and ProcessTable classes.
// These tests should print out appropriate diagnostics that you can
// use to verify that your classes are implemented correctly.

void printResult(int testNumber,
                string expected_value,
                string actual_value)
{
    cout << "Test " << testNumber;

    if (expected_value == actual_value)
    {
        cout << " PASS: ";
    } else {
        cout << " FAIL: ";
    }

    cout << "Expected: [" << expected_value << "] ";
    cout << "Actual: [" << actual_value << "]" << endl << endl;
}


int 
main(int argc, char ** argv)
{
    // Add your tests here. Make sure to document the expected behavior for
    // each test case and print the results. You may wish to use 
    // the printing capabilities of your classes as part of your
    // testing.

    int testNum = 1;


    //
    // These are tests for the base Process class.
    //

    // Using the default constructor, the process name should be empty, and the process ID
    // should be -1
    Process p1;
    ostringstream s;
    s << p1;
    printResult(testNum++, R"({"cpu_needed":0,"cpu_used":0,"priority":100,"process_id":-1,"process_name":""})", s.str());


    // Set up the process using the constructor
    Process p2("name2", 100, 0, 2);
    s.str("");
    s.clear();
    s << p2;
    printResult(testNum++, R"({"cpu_needed":0,"cpu_used":0,"priority":100,"process_id":2,"process_name":"name2"})", s.str());

    // Test the setters 
    Process p3;
    s.str("");
    s.clear();

    p3.setProcessName("name3");
    p3.setProcessId(3);
    p3.setPriority(10);
    p3.setCpuNeeded(9);
    s << p3;
    printResult(testNum++, R"({"cpu_needed":9,"cpu_used":0,"priority":10,"process_id":3,"process_name":"name3"})", s.str());


    // Test the getters
    Process p4("name4", 22, 11, 4);
    s.str("");
    s.clear();

    s << "Name: " << p4.processName() << " Id: " << p4.processId() << " Priority: " << p4.priority()
      << " CPU Needed: " << p4.cpuNeeded() << " CPU Used: " << p4.cpuUsed();
    printResult(testNum++, "Name: name4 Id: 4 Priority: 22 CPU Needed: 11 CPU Used: 0", s.str());


    //
    // These are tests for the ProcessTable class.
    // We'll use the p1 through p4 variables defined above, for convenience.
    //

    p1.setProcessName("name1");

    ProcessTable t1;

    t1.add(p1);
    t1.add(p2);
    t1.add(p3);
    t1.add(p4);
    
    s.str("");
    s.clear();
    s << t1;

    string exp = R"({"cpu_needed":0,"cpu_used":0,"priority":100,"process_id":1,"process_name":"name1"}
{"cpu_needed":0,"cpu_used":0,"priority":100,"process_id":2,"process_name":"name2"}
{"cpu_needed":9,"cpu_used":0,"priority":10,"process_id":3,"process_name":"name3"}
{"cpu_needed":11,"cpu_used":0,"priority":22,"process_id":4,"process_name":"name4"}
)";

    printResult(testNum++, exp, s.str());
            

    // Now test find()
    Process found = t1.find(2);
    int rc;

    s.str(""); 
    s.clear();
    s << found;
    printResult(testNum++, R"({"cpu_needed":0,"cpu_used":0,"priority":100,"process_id":2,"process_name":"name2"})", s.str());

    // Test a find for a nonexistent process
    found = t1.find(5);
    s.str("");
    s.clear();
    s << found;
    printResult(testNum++, R"({"cpu_needed":0,"cpu_used":0,"priority":0,"process_id":-1,"process_name":""})", s.str());

    // Now test remove(). 
    // We'll remove process Id 3, and expect it not to print.
    rc = t1.remove(3);
    s.str("");
    s.clear();
    s << "rc: " << rc << endl << t1;
    
    exp = R"(rc: 0
{"cpu_needed":0,"cpu_used":0,"priority":100,"process_id":1,"process_name":"name1"}
{"cpu_needed":0,"cpu_used":0,"priority":100,"process_id":2,"process_name":"name2"}
{"cpu_needed":11,"cpu_used":0,"priority":22,"process_id":4,"process_name":"name4"}
)";

    printResult(testNum++, exp, s.str());


    // Now we'll use a smaller table, and test the wraparound stuff.

    // This table will only hold 3 entries.
    ProcessTable t2(3);

    t2.add(p1);
    t2.add(p2);
    t2.add(p3);

    // Now print it out.
    s.str("");
    s.clear();
    s << t2;
    
    exp = R"({"cpu_needed":0,"cpu_used":0,"priority":100,"process_id":1,"process_name":"name1"}
{"cpu_needed":0,"cpu_used":0,"priority":100,"process_id":2,"process_name":"name2"}
{"cpu_needed":9,"cpu_used":0,"priority":10,"process_id":3,"process_name":"name3"}
)";
    printResult(testNum++, exp, s.str());


    // Now try to add the fourth process - it should fail
    rc = t2.add(p4);
    s.str("");
    s.clear();
    s << "rc: " << rc;

    printResult(testNum++, "rc: -1", s.str());

    // We will now remove process, and then add our new process there.
    t2.remove(2);
    t2.add(p4);
    
    s.str("");
    s.clear();
    s << t2;
    
    exp = R"({"cpu_needed":0,"cpu_used":0,"priority":100,"process_id":1,"process_name":"name1"}
{"cpu_needed":11,"cpu_used":0,"priority":22,"process_id":2,"process_name":"name4"}
{"cpu_needed":9,"cpu_used":0,"priority":10,"process_id":3,"process_name":"name3"}
)";

    printResult(testNum++, exp, s.str());


    // Now test the 'run' method. 

    Process p5("name5", 10, 4, 5);

    // Before we run, it should show 0 used
    s.str("");
    s.clear();
    s << p5;
    printResult(testNum++, R"({"cpu_needed":4,"cpu_used":0,"priority":10,"process_id":5,"process_name":"name5"})", s.str());

    // Now we run it for 1 cycle - we expect it to return 1 for the number to run,
    // and update the internals 

    rc = p5.run(1);
    s.str("");
    s.clear();
    s << "rc: " << rc << endl << p5;

    printResult(testNum++, 
    R"(rc: 1
{"cpu_needed":4,"cpu_used":1,"priority":10,"process_id":5,"process_name":"name5"})", s.str());


    // Now run it for 2 cycles.
    rc = p5.run(2);
    s.str("");
    s.clear();
    s << "rc: " << rc << endl << p5;

    printResult(testNum++, 
    R"(rc: 2
{"cpu_needed":4,"cpu_used":3,"priority":10,"process_id":5,"process_name":"name5"})", s.str());

    // Now try to run it for 2 more cycles, but it should only actually do 1
     rc = p5.run(2);
    s.str("");
    s.clear();
    s << "rc: " << rc << endl << p5;

    printResult(testNum++, 
    R"(rc: 1
{"cpu_needed":4,"cpu_used":4,"priority":10,"process_id":5,"process_name":"name5"})", s.str());

    // And now ask it to run anything, and it should just return 0
    // Now try to run it for 2 more cycles, but it should only actually do 1
     rc = p5.run(1000);
    s.str("");
    s.clear();
    s << "rc: " << rc << endl << p5;

    printResult(testNum++, 
    R"(rc: 0
{"cpu_needed":4,"cpu_used":4,"priority":10,"process_id":5,"process_name":"name5"})", s.str());

}
