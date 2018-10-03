/*
 * For the first part of this project, my goal was to implement an expression tree. After constructing the ExpressionTree, I provided
 * different methods that return different ouput when invoked on an expression tree. ExpressionTreeInstantiate.java instantiates an
 * expression tree on a hard coded string representing a postfix expression tree and demonstrates all of the aforementioned methods. 
 *
 * For the second part of this project, my goal was to index text with an AVL tree. The program goes through the input file line by
 * line, extracting each word, and inserting that word, along with it's number line to the AVL tree. The file being indexed then gets
 * passed into the program as a command line argument. There are various methods involved for indexing. When indexing is complete,
 * the program calls upon a method to display a list of unique words in teh text file and the line numbers in which that word occurs.
 */  

/*
 * Part 1
 */

/*
 * ExpressionTree.java
 */

// Represents an expression tree
public class ExpressionTree {

    private ExpressionNode root;

    // creates anew expression tree from the postfix expression
    public ExpressionTree(String postfix) {
	// create empty stack
	MyStack<ExpressionNode> stack = new MyStack<>();

	// split into tokens
	String[] tokens = postfix.split("[ ]+");

	for (int i = 0; i < tokens.length; i++) {
	    // get token
	    String token = tokens[i];

	    if (isOperator(token)) { // if operator
		// pop and create a new node
		ExpressionNode right = stack.pop();
		ExpressionNode left = stack.pop();
		ExpressionNode newNode = new ExpressionNode(token.charAt(0), left, right);
		stack.push(newNode); // add to stack
	    } else { // if operand
		ExpressionNode newNode = new ExpressionNode(Integer.parseInt(token));
		stack.push(newNode);
	    }
	}

	// last node int stack is the root of tree
	root = stack.pop();
    }

    // check if the token is an opertor
    private boolean isOperator(String s) {
	return s.equals("+") || s.equals("-") || s.equals("*") || s.equals("/");
    }

    // evaluate value of expression
    public int eval() {
	return eval(root);
    }

    // helper method to evaluate the expression with the current node
    private int eval(ExpressionNode current) {
	if (current.isDataNode()) // data node
	    return current.data;
	else
	    return eval(eval(current.left), current.operator, eval(current.right));
    }

    // evaluate a operator and two operands
    private int eval(int x, char op, int y) {
	int result = 0;
	switch (op) {
	case '+':
	    result = x + y;
	    break;
	case '-':
	    result = x - y;
	    break;
	case '*':
	    result = x * y;
	    break;
	case '/':
	    result = x / y;
	    break;

	default:
	    break;
	}

	return result;
    }

    // postfix expression
    public String postfix() {
	return postfix(root);
    }

    // helper method for postfix expression
    private String postfix(ExpressionNode current) {
	if (current.isDataNode())
	    return current.data + "";
	else
	    return postfix(current.left) + " " + postfix(current.right) + " " + current.operator;
    }

    // prefix expression
    public String prefix() {
	return prefix(root);
    }

    // helper method for prefix expression
    private String prefix(ExpressionNode current) {
	if (current.isDataNode())
	    return current.data + "";
	else
	    return current.operator + " " + prefix(current.left) + " " + prefix(current.right);
    }

    // infix expression
    public String infix() {
	return infix(root);
    }

    
    // helper method for infix expression 
    private String infix(ExpressionNode current) {
	if (current.isDataNode())
	    return current.data + "";
	else
	    return "(" + infix(current.left) + " " + current.operator + " " + infix(current.right) + ")";
    }

    
    // represent a expression tree's node
    private class ExpressionNode {
	// can have data or oprator
	private int data;
	private char operator;
	private ExpressionNode left, right;

	
	// operator tree node
	public ExpressionNode(char operator, ExpressionNode left, ExpressionNode right) {
	    this.operator = operator;
	    this.left = left;
	    this.right = right;
	}

	
	// data node has no children
	public ExpressionNode(int data) {
	    this.data = data;
	}

	
	// data note has no children
	public boolean isDataNode() {
	    return left == null && right == null;
	}

    }
}

/*
 * MyStack.java
 */

import java.util.LinkedList;

public class MyStack<AnyType>{
    
    private LinkedList<AnyType> stack;
    
    
    public MyStack() {
	stack = new LinkedList<>();
    }
    
    public void push(AnyType x){
	stack.addFirst(x); //add at start
    }
    
    public AnyType pop() {
	if(isEmpty()) // if empty
	    return null;
	
	return stack.removeFirst();
    }
    
    public AnyType peek() {
	if(isEmpty()) // if empty
	    return null;
	
	return stack.getFirst();
    }
    
    
    // check if stack is empty
    public boolean isEmpty() {
	return stack.size() == 0;
    }
    
    // number of element in stack
    public int size(){
	return stack.size();
    }
}

/*
 * ExpressionTreeInstantiate.java
 */

public class ExpressionTreeInstantiate {
    public static void main(String[] args) {
	String postfix = "34 2 - 5 *";

	ExpressionTree tree = new ExpressionTree(postfix);
	
	System.out.println("Evaluate: " + tree.eval());
	System.out.println("Postfix: " + tree.postfix());
	System.out.println("Prefix: " + tree.prefix());
	System.out.println("Infix: " + tree.infix());

    }
}


/*
 * Part 2
 */

import java.util.LinkedList;
import java.util.List;

// AvlTree class

/**
 * Implements an AVL tree. Note that all "matching" is based on the compareTo
 * method.
 * 
 * @author Mark Allen Weiss
 */
public class AvlTree {
    public AvlTree() {
	root = null;
    }

    public void indexWord(String x, int line) {
	root = insert(x, line, root);
    }
   
    public void printIndex() {
	if (isEmpty())
	    System.out.println("Empty tree");
	else
	    printIndex(root);
    }

    private boolean isEmpty() {
	return root == null;
    }

    public List<Integer> getLinesForWord(String word) {
	return getLinesForWord(word, root);
    }

    private static final int ALLOWED_IMBALANCE = 1;

    private AvlNode balance(AvlNode t) {
	if (t == null)
	    return t;

	if (height(t.left) - height(t.right) > ALLOWED_IMBALANCE)
	    if (height(t.left.left) >= height(t.left.right))
		t = rotateWithLeftChild(t);
	    else
		t = doubleWithLeftChild(t);
	else if (height(t.right) - height(t.left) > ALLOWED_IMBALANCE)
	    if (height(t.right.right) >= height(t.right.left))
		t = rotateWithRightChild(t);
	    else
		t = doubleWithRightChild(t);

	t.height = Math.max(height(t.left), height(t.right)) + 1;
	return t;
    }
    
    private AvlNode insert(String x, int line, AvlNode t) {
	if (t == null)
	    return new AvlNode(x, null, null, line);

	int compareResult = x.compareTo(t.element);

	if (compareResult < 0)
	    t.left = insert(x, line, t.left);
	else if (compareResult > 0)
	    t.right = insert(x, line, t.right);
	else if (!t.indexes.contains(line))
	    t.indexes.add(line);
	return balance(t);
    }

    private List<Integer> getLinesForWord(String x, AvlNode t) {
	while (t != null) {
	    int compareResult = x.compareTo(t.element);

	    if (compareResult < 0)
		t = t.left;
	    else if (compareResult > 0)
		t = t.right;
	    else
		return t.indexes; // Match
	}

	return null; // No match
    }

    private void printIndex(AvlNode t) {
	if (t != null) {
	    printIndex(t.left);
	    System.out.println(t.element + " " + t.indexes);
	    printIndex(t.right);
	}
    }

    private int height(AvlNode t) {
	return t == null ? -1 : t.height;
    }

    private AvlNode rotateWithLeftChild(AvlNode k2) {
	AvlNode k1 = k2.left;
	k2.left = k1.right;
	k1.right = k2;
	k2.height = Math.max(height(k2.left), height(k2.right)) + 1;
	k1.height = Math.max(height(k1.left), k2.height) + 1;
	return k1;
    }

    private AvlNode rotateWithRightChild(AvlNode k1) {
	AvlNode k2 = k1.right;
	k1.right = k2.left;
	k2.left = k1;
	k1.height = Math.max(height(k1.left), height(k1.right)) + 1;
	k2.height = Math.max(height(k2.right), k1.height) + 1;
	return k2;
    }

    private AvlNode doubleWithLeftChild(AvlNode k3) {
	k3.left = rotateWithRightChild(k3.left);
	return rotateWithLeftChild(k3);
    }

    private AvlNode doubleWithRightChild(AvlNode k1) {
	k1.right = rotateWithLeftChild(k1.right);
	return rotateWithRightChild(k1);
    }

    private static class AvlNode {

	AvlNode(String theElement, AvlNode lt, AvlNode rt, int index) {
	    element = theElement;
	    left = lt;
	    right = rt;
	    height = 0;
	    indexes = new LinkedList<>();
	    indexes.add(index);
	}

	String element; // The data in the node
	AvlNode left; // Left child
	AvlNode right; // Right child
	int height; // Height
	LinkedList<Integer> indexes; // list of line numbers in which the word occurs
    }

    private AvlNode root;

}

/*
 * Part2.java
 */

public class Part2{
    public static void main(String[] args) {
	String fileName = args[0];
	
	// create tree
	AvlTree tree = new AvlTree();
	
	try {
	    
	    // to read lines
	    Scanner scanner = new Scanner(new File(fileName));
	    
	    // read line by line
	    int lineNum = 0;
	    while(scanner.hasNextLine()) {
		lineNum++;
		String line = scanner.nextLine();
		// split line
		String[] words = line.split("[ ]+");
		
		// words
		for (String word : words) {
		    // remove punctuation and ignore case
		    word = word.replaceAll("[^a-zA-Z0-9]", "").toLowerCase();
		    tree.indexWord(word, lineNum); // add
		}
		   
	    }
	    scanner.close();
	    
	    tree.printIndex();
	    
	} catch (FileNotFoundException e) {
	    e.printStackTrace();
	}
	
    }
}

/*
 * UnderflowException.java
 */

/**
 * Exception class for access in empty containers
 * such as stacks, queues, and priority queues.
 * @author Mark Allen Weiss
 */
public class UnderflowException extends RuntimeException
{
}
