/*
 * For the first part of this project, I had to write a program that outputs whether or not symbols (input) are appropriately balanced and if not, indicate which error condition occurred and what symbol type caused the problem. I wrote MyStack.java and SymbolBalance.java. In the implementation of my stack methods, I only used basic list operations, not the stack operations themselves.
 *
 * For the second part of this project, I built a queue out of two seperate stacks, S1 and S2. Enqueue operations happened by pushing data onto stack 1 while dequeue operations were completed with a pop from stack 2. I wrote TwoSTackQueue.java that provides the Queue ADT using two stacks and a tester class with a main method ot test out TwoStackQueue. I also discussed the big-O running time of the enqueue and dequeue operation.
 */


/* 
 * MyStack.java
 */


import java.util.LinkedList;

// Represents a stack
public class MyStack<AnyType>{
    
    // backing data straucture to implemnt stack
    private LinkedList<AnyType> stack;
    
    
    // creates emptys tack
    public MyStack() {
	stack = new LinkedList<>();
    }
    
    
    // push to stack
    public void push(AnyType x){
	stack.addFirst(x); //add at start
    }
    
    
    // pop from stack
    public AnyType pop() {
	if(isEmpty()) // if empty
	    return null;
	
	// remove first
	return stack.removeFirst();
    }
    
    
    // peek from stack
    public AnyType peek() {
	if(isEmpty()) // if empty
	    return null;
	
	return stack.getFirst();
    }
    
    
    // chekc if stack is empty
    public boolean isEmpty() {
	return stack.size() == 0;
    }
    
    // number of element in stack
    public int size(){
	return stack.size();
    }
    
}

/*
 * SymbolBalance.java
 */

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class SymbolBalance {

    public static void main(String[] args) {
	String file = args[0];

	// create a stack to store symbols
	MyStack<Character> stack = new MyStack<>();

	try {
	    // open scanner to read from file
	    Scanner scanner = new Scanner(new File(file));

	    // whether the file is valid or not
	    boolean valid = true;

	    // whether the comment block is started or not
	    boolean commentStarted = false;

	    // while the file is valid and there are more lines to read
	    while (scanner.hasNextLine() && valid) {

		// read line
		String line = scanner.nextLine();

		// remove extra spaces
		line = line.trim();

		// remove quotes and anything between them
		line = removeQuotes(line);

		// current index
		int iCurrent = 0;

		// while there are indexes left to explore and the file is till
		// valid
		while (iCurrent < line.length() && valid) {

		    // get teh current character
		    char current = line.charAt(iCurrent);

		    // if comment is not started yet and its /
		    if (!commentStarted && current == '/') {

			// check if its opening
			if (iCurrent + 1 < line.length() && line.charAt(iCurrent + 1) == '*') {
			    commentStarted = true; // set opening
			    iCurrent++; // skip the character
			}
		    } else if (current == '*') { // if start found

			// check if end found
			if (iCurrent + 1 < line.length() && line.charAt(iCurrent + 1) == '/') {
			    if (commentStarted == true) { // if there was a
							  // start
				commentStarted = false;
				iCurrent++;
			    } else { // no start comment
				printUnbalanced("*/");
				valid = false;
			    }

			}
		    }

		    // if comment hasnt started and quote found
		    else if (!commentStarted && current == '"') {
			printUnbalanced(current);
			valid = false;
		    } else if (!commentStarted && isOpening(current)) { // if
									// opening
			stack.push(current); // push to stack
		    } else if (!commentStarted && isClosing(current)) { // if
									// closing
			if (stack.isEmpty()) { // if empty
			    printUnbalanced(current);
			    valid = false;
			} else {
			    // remove top
			    char popped = stack.pop();
			    if (!match(popped, current)) { // if not matching
				printUnbalanced(current);
				valid = false;
			    }
			}
		    }

		    // move to next character
		    iCurrent++;
		}

	    }

	    // if in the end it was valid
	    if (valid) {
		if (commentStarted) // check if comment is started and not
				    // finished
		    printUnbalanced("/*");
		else if (!stack.isEmpty()) // check if any symbol is unmacthed
		    printUnbalanced(stack.pop());
		else // all okay
		    System.out.println("Balanced");
	    }
	    
	    scanner.close();

	} catch (FileNotFoundException e) {
	    e.printStackTrace();
	}
    }

    // check if a character is opening bracket
    public static boolean isOpening(char c) {
	return c == '{' || c == '[' || c == '(';
    }

    // check if a character is closing bracket
    public static boolean isClosing(char c) {
	return c == '}' || c == ']' || c == ')';
    }

    // check if a opening and closing brackets match
    public static boolean match(char opening, char closing) {
	return (opening == '{' && closing == '}') || (opening == '(' && closing == ')')
		|| (opening == '[' && closing == ']');
    }

    
    // print unbalanced error message
    
    public static void printUnbalanced(char mismatched) {
	printUnbalanced(mismatched + "");
    }

    public static void printUnbalanced(String mismatched) {
	System.out.println("Unbalanced! Symbol " + mismatched + " is mismatched.");
    }

    
    // removes quotes and anything between them from the line
    public static String removeQuotes(String line) {
	while (true) {
	    // get start and end indices
	    int iStart = line.indexOf('"');
	    int iEnd = line.indexOf('"', iStart + 1);
	    if (iStart == -1 || iEnd == -1) // if no start or end
		break;
	    else
		// remove start and end
		line = line.substring(0, iStart) + line.substring(iEnd + 1);
	}

	return line;
    }

}


/*
 * TwoStackQueue.java
 */


public class TwoStackQueue<AnyType> implements MyQueue<AnyType>{
    
    // two stacks for each stack
    private MyStack<AnyType> stackA, stackB;
    
    
    // creates empty queue 
    public TwoStackQueue() {
	// initialise empty stacks
	stackA = new MyStack<>();
	stackB = new MyStack<>();
    }

    @Override
    public void enqueue(AnyType x) {
	stackA.push(x); // all inserts are done in stack A
	
    }

    @Override
    public AnyType dequeue() {
	if(stackB.isEmpty()) { // stack B is empty
	    while(!stackA.isEmpty()) { // move elements from stack A to B
		stackB.push(stackA.pop());
	    }
	}
	
	// remove from stack B
	return stackB.pop();
    }

    @Override
    public boolean isEmpty() {
	//check both emtpty
	return stackA.isEmpty() && stackB.isEmpty();
    }

    @Override
    public int size() {
	// add size of both stacks
	return stackA.size() + stackB.size();
    }

}

/*
 * MyQueue.java
 */

public interface MyQueue<AnyType> {

    // Performs the enqueue operation
    public void enqueue(AnyType x);

    // Performs the dequeue operation. For this assignment, if you
    // attempt to dequeue an already empty queue, you should return
    // null
    public AnyType dequeue();

    // Checks if the Queue is empty
    public boolean isEmpty();

    // Returns the number of elements currently in the queue
    public int size();

}

/*
 * TesterForTwoStackQueue.java
 */

public class Problem2{
    
    public static void main(String[] args) {
	// create queue
	MyQueue<String> queue = new TwoStackQueue<>();
	
	// print size and empty status
	System.out.println("Size [Expected 0]: " +queue.size());
	System.out.println("Empty? " +queue.isEmpty());
	
	// Add and print size after each insert
	
	System.out.println("\nAdding A, B, C and printing size after each insert...");
	queue.enqueue("A");
	System.out.println(queue.size());
	
	queue.enqueue("B");
	System.out.println(queue.size());
	
	queue.enqueue("C");
	System.out.println(queue.size());
	
	
	// dequeue and rint status
	
	System.out.println("\nDequeue [Expected: A]: "+queue.dequeue());
	System.out.println("Size [Expected 2]: " +queue.size());
	System.out.println("Empty? " +queue.isEmpty());
	
	System.out.println("\nDequeue [Expected: B]: "+queue.dequeue());
	System.out.println("Size [Expected 1]: " +queue.size());
	System.out.println("Empty? " +queue.isEmpty());
	
	// enqueue again
	
	System.out.println("\nEnqueue D");
	queue.enqueue("D");
	System.out.println("Size [Expected 2]: " +queue.size());
	System.out.println("Empty? " +queue.isEmpty());
	
	
	// dequeue again and print status
	
	
	System.out.println("\nDequeue [Expected: C]: "+queue.dequeue());
	System.out.println("Size [Expected 1]: " +queue.size());
	System.out.println("Empty? " +queue.isEmpty());
	
	System.out.println("\nDequeue [Expected: D]: "+queue.dequeue());
	System.out.println("Size [Expected 0]: " +queue.size());
	System.out.println("Empty? " +queue.isEmpty());
	
	
	
	
    }
}

/*
 * Time Complexity - big-O running time of enqueue and dequeue operation for queue implementation
 */


Enqueue: O(1)
Explanation: Enqueue uses only one stack. It pushes the element to top of stack. Since time complexity of push in stack
is O(1), the overall operation also costs O(1)


Dequeue: O(n)
Explanation: The dequeue operation on first dequeue will cost O(n) time since all the elements are shifted from stack
A to stack B. However after that, until the stack B becomes empty it costs only O(1) time as the element
is popped directly from the stack 2.

