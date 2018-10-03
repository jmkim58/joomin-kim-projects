/*
 * For this project, my goal was to implement Dijkstra's algorithm to find shortest paths between pairs of cities on a map. 
 I modified Dijkstra.java, a graph class to find the shortest path.
 Files Vertex.java, Edge.java, Display.java (displays graph in GUI), and citypairs.txt for the text input files were provided.
 */

/*
 * Dijkstra.java
 */


import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.io.IOException;
import java.io.FileReader;
import java.io.BufferedReader;

public class Dijkstra {

    private Map<String, Vertex> vertexNames;

    public Dijkstra() {
	vertexNames = new HashMap<String, Vertex>();
    }

    public void addVertex(Vertex v) {
	if (vertexNames.containsKey(v.name))
	    throw new IllegalArgumentException("Cannot create new vertex with existing name.");
	vertexNames.put(v.name, v);
    }
	
    public Collection<Vertex> getVertices() {
	return vertexNames.values();
    }

    public Vertex getVertex(String name) {
	return vertexNames.get(name);
    }

    public void addEdge(String nameU, String nameV, Double cost) {
	if (!vertexNames.containsKey(nameU))
	    throw new IllegalArgumentException(nameU + " does not exist. Cannot create edge.");
	if (!vertexNames.containsKey(nameV))
	    throw new IllegalArgumentException(nameV + " does not exist. Cannot create edge.");
	Vertex sourceVertex = vertexNames.get(nameU);
	Vertex targetVertex = vertexNames.get(nameV);
	Edge newEdge = new Edge(sourceVertex, targetVertex, cost);
	sourceVertex.addEdge(newEdge);
    }

    public void addUndirectedEdge(String nameU, String nameV, double cost) {
	addEdge(nameU, nameV, cost);
	addEdge(nameV, nameU, cost);
    }

    // MY CODE STARTS HERE

    public double computeEuclideanDistance(double ux, double uy, double vx, double vy) {
	return Math.sqrt(Math.pow(vx - ux, 2) + Math.pow(vy - uy, 2));
    }

    public void computeAllEuclideanDistances() {
	for (Vertex v : vertexNames.values()) {
	    for (Edge e : v.adjacentEdges) {
		e.distance = computeEuclideanDistance(e.source.x, e.source.y, e.target.x, e.target.y);
	    }
	}
    }

    public void doDijkstra(String s) {
	// define infinity
	final double INFINITY = Double.MAX_VALUE;

	
	// Queue for vertices
	ArrayList<Vertex> Q = new ArrayList<>();

	
	// for each vertex
	for (Vertex v : vertexNames.values()) {
	    v.known = false; // not known
	    v.distance = INFINITY; // set distance
	    v.prev = null; // no prev
	    Q.add(v); // add to list
	}

	
	// visit source
	Vertex source = vertexNames.get(s);
	source.distance = 0;

	while (!Q.isEmpty()) {
	    
	    // remove min vertex
	    Vertex u = removeMin(Q);
	    
	    // set as known
	    u.known = true;

	    for (Edge e : u.adjacentEdges) {
		// get target
		Vertex v = e.target;

		
		// if not known
		if (!v.known) {
		    //calc alternative distance
		    double alt = u.distance + e.distance;
		    
		    // if better
		    if (alt < v.distance) {
			v.distance = alt;
			v.prev = u;
		    }
		}

	    }
	}

    }

    private Vertex removeMin(ArrayList<Vertex> Q) {
	Vertex min = Q.get(0);
	for (int i = 1; i < Q.size(); i++) {
	    Vertex curr = Q.get(i);
	    if (curr.distance < min.distance)
		min = curr;
	}

	Q.remove(min);

	return min;
    }

    public List<Edge> getDijkstraPath(String s, String t) {
	doDijkstra(s);

	
	// holds sequence of vertices
	List<Vertex> list = new ArrayList<>();
	
	
	// get target
	Vertex u = vertexNames.get(t);
	
	// go back until not at start
	while(u.prev != null) {
	    list.add(0, u);
	    u = u.prev;
	}
	list.add(0, u);
	
	// find edges from sequence of vertices
	List<Edge> path = new ArrayList<>();
	for (int i = 1; i < list.size(); i++) {
	    Vertex source = list.get(i-1);
	    Vertex target = list.get(i);
	    for (Edge edge : source.adjacentEdges) {
		if(edge.target == target) {
		    path.add(edge);
		}
	    }
	}
	
	return path;
	
    }

    // MY CODE ENDS HERE


    public void printAdjacencyList() {
	for (String u : vertexNames.keySet()) {
	    StringBuilder sb = new StringBuilder();
	    sb.append(u);
	    sb.append(" -> [ ");
	    for (Edge e : vertexNames.get(u).adjacentEdges) {
		sb.append(e.target.name);
		sb.append("(");
		sb.append(e.distance);
		sb.append(") ");
	    }
	    sb.append("]");
	    System.out.println(sb.toString());
	}
    }

    public static void main(String[] argv) throws IOException {
	String vertexFile = "cityxy.txt";
	String edgeFile = "citypairs.txt";

	Dijkstra dijkstra = new Dijkstra();
	String line;

	// Read in the vertices
	BufferedReader vertexFileBr = new BufferedReader(new FileReader(vertexFile));
	while ((line = vertexFileBr.readLine()) != null) {
	    String[] parts = line.split(",");
	    if (parts.length != 3) {
		vertexFileBr.close();
		throw new IOException("Invalid line in vertex file " + line);
	    }
	    String cityname = parts[0];
	    int x = Integer.valueOf(parts[1]);
	    int y = Integer.valueOf(parts[2]);
	    Vertex vertex = new Vertex(cityname, x, y);
	    dijkstra.addVertex(vertex);
	}
	vertexFileBr.close();

	BufferedReader edgeFileBr = new BufferedReader(new FileReader(edgeFile));
	while ((line = edgeFileBr.readLine()) != null) {
	    String[] parts = line.split(",");
	    if (parts.length != 3) {
		edgeFileBr.close();
		throw new IOException("Invalid line in edge file " + line);
	    }
	    dijkstra.addUndirectedEdge(parts[0], parts[1], Double.parseDouble(parts[2]));
	}
	edgeFileBr.close();

	dijkstra.computeAllEuclideanDistances();

	dijkstra.printAdjacencyList();

	String startCity = "SanFrancisco";
	String endCity = "Boston";

	List<Edge> path = dijkstra.getDijkstraPath(startCity, endCity);

	System.out.print("Shortest path between " + startCity + " and " + endCity + ": ");
	System.out.println(path);
    }

}

/*
 * Vertex.java
 */

import java.util.LinkedList;
import java.util.List;

public class Vertex {

    public String name;
    public int x;
    public int y;
    public boolean known;
    public double distance; // total distance from origin point
    public Vertex prev;
    public List<Edge> adjacentEdges;

    public Vertex(String name, int x, int y) {
	this.name = name;
	this.x = x;
	this.y = y;
	// by default java sets uninitialized boolean to false and double to 0
	// hence known == false and dist == 0.0
	adjacentEdges = new LinkedList<Edge>();
	prev = null;
    }

    @Override
    public int hashCode() {
	// we assume that each vertex has a unique name
	return name.hashCode();
    }

    @Override
    public boolean equals(Object o) {
	if (this == o) {
	    return true;
	}
	if (o == null) {
	    return false;
	}
	if (!(o instanceof Vertex)) {
	    return false;
	}
	Vertex oVertex = (Vertex) o;

	return name.equals(oVertex.name) && x == oVertex.x && y == oVertex.y;
    }

    public void addEdge(Edge edge) {
	adjacentEdges.add(edge);
    }

    public String toString() {
	return name + " (" + x + ", " + y + ")";
    }

}

/*
 * Edge.java
 */

public class Edge {

    public double distance;
    public Vertex source;
    public Vertex target;

    public Edge(Vertex vertex1, Vertex vertex2, double weight) {
	source = vertex1;
	target = vertex2;
	this.distance = weight;
    }

    public String toString() {
	return source + " - " + target;
    }
}
