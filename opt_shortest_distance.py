
from typing import Generic
from typing import TypeVar, Generic
T = TypeVar('T')


from typing import TypeVar, Generic

T = TypeVar('T')
class MinHeap(Generic[T]):

    def __init__(self, max_size) -> None:
        """
        Function description: Constructor of min heap
        Approach description:
            self.index_array holds the position (index) of vertex object in self.the_array.
            In other words,
            self.index_array - index: Vertex id
                             - value: index of that vertex id "IN self.the_array"
            Therefore, Once I get vertex id, I can straightly get the position or index of that vertex in self.the_array
            When I add vertex, or swap vertices, change the values accordingly as well.
        Input: 
             max_size: integer of the size for the min heap(max capacity for min heap)
        Time complexity: O(T) where T is max size (or max id of vertices or trees) .
            time complexity analysis: creating index array takes O(T)  where T is the set of unique trees.
        Space complexity: O(T) where T is max size (or max id of vertices or trees) .
            space complexity analysis: 
                max_size will be the double of T, so 2T when escape function is called
                For future use of escape function, I initially give 2T to max_size as input in escape function.
                However, O(2T) = O(T)
            
        """
        self.length = 0
        self.the_array = [None]
        self.index_array = [-1] * (max_size + 1)
        
    def __len__(self)-> None:
        """
        Function description:
            Get the number of items in the heap(Except for None)
        Time complexity: O(1)
        Aux Space complexity: O(1)
        """
        return self.length
        

    def rise(self, k: int) -> None:
        """
        Function description:
            Comparing with a parent node, rise element at index k to its correct position
        Time Complexity: O(log V) where V is the number of the items in the heap
        Aux Space Complexity: O(1)
        Input: An index of integer
        """
        parent = k // 2
        while parent >= 1:
            if self.the_array[parent][1] > self.the_array[k][1]: # As long as "time" of parent node is bigger, keep swapping
                self.swap(parent, k)
                k = parent
                parent = k // 2 # get index of parent node from one of its child
            else:
                break

    def add(self, element: T) -> bool:
        """
        Function description: 
            Add a element or item in min heap.
        Time Complexity: O(log V), where V is the number of items in the MinHeap
        Aux Space Complexity: O(1)
        Input: 
            element: item of (tree obj, time), time is integer
        """
        self.the_array.append(element) 
        self.length += 1
        index = element[0].id # get the index of vertex in self.the_array
        #add index position info of added vertex in self.index_array
        self.index_array[index] = self.length
        self.rise(self.length)

    def sink(self, k: int) -> None:
        """ 
        Function description:
            Make the element at index k sink to the correct position.
        Time complexity: O(log V) where V is the number of items in the MinHeap
        Aux Space complexity: O(1)
        Input: index of integer
        """
        child = 2 * k
        while child <= self.length:
            if child < self.length and self.the_array[child+1][1] < self.the_array[child][1]:
                child+=1
            if self.the_array[k][1] > self.the_array[child][1]:
                self.swap(k, child)
                k = child
                child = 2 * k
            else:
                break # loop terminate at the correct position

    def serve(self):
        """ 
        Function description:
            Remove and return the smallest element(smallest amount of time) from the heap. 
        Time complexity: O(log V), where V is the number of items in a heap
        Aux Space complexity: O(1)
        Output: item of (tree obj, time)
        """
        self.swap(1, self.length)
        self.length -= 1
        self.sink(1) # swapped elem now is moving at the correct position

        return self.the_array.pop()[0] # serve the item with min time

    def update(self, vertex, time):
        """
        Function description:
            Update the time of vertex given in a input.
        Approach description:
            self.index_array holds the position (index) of vertex object in self.the_array.
            In other words,
            self.index_array - index: Vertex id
                             - value: index or position of that vertex id IN self.the_array
            Therefore, Once I get vertex id, I can straightly get the position or index of that vertex in self.the_array and I can change the time in self.the_array
        Input:
            vertex: vertex object(tree obj) I want to update the time
            time: An index of the integer represents amount of time to take from the source.
        Time Complexity: O(1)  where V is the number of items in the MinHeap
            Time complexity analysis:
                As I mentioned in Approach description, I can access to (vertex, time) item in the array only by index access.
        Aux Space Complexity: O(1)
        
        """
        index = self.index_array[vertex.id]
        self.the_array[index] = (vertex, time) # reassign the updated time item

    def swap(self,a, b):
        """
        Function description: Swap the positions in array
        Time Complexity: O(1)
        Aux Space Complexity: O(1)
        Input:
            a: An index of the integer 
            b: An index of the integer
        """
        self.the_array[a], self.the_array[b] = self.the_array[b], self.the_array[a]
        self.index_array[self.the_array[a][0].id],self.index_array[self.the_array[b][0].id] = self.index_array[self.the_array[b][0].id], self.index_array[self.the_array[a][0].id]
        #swap the index position for swapped vertices as well



class TreeMap:
    def __init__(self, roads, solulus) -> None: # V: take a list of names of vertices as numbers
        """
        Function description: 
            This is the constructor for Tree map/graph.
            This function takes input of information of roads and solulu and make graph(Tree map).
        Approach description:
            Since roads (input) contains info of  vertices id, and the id always start from 0,
            you can calculate the number of uniq vertices from checking their max id of vertices.
            Then, instantiate vertices and edges manually and store them in appropriate variables.
        Time complexity: O(T+R) where R is the set of roads and T is the set of unique trees
            Time complexity analysis: 
                Creating array(Size T) and initializing by vertices objects takes O(T)
                getting the uniq number of trees and adding edges takes O(R)
                Therefore in total, O(T+R)
        Space Complexity: O(T+R) where R is the set of roads and T is the set of unique trees
            Input Space complexity analysis: Input takes roads and solulus, which take O(R+T)
            Aux Space complexity analysis: Array of vertices(trees) takes O(T) Aux space
        Input: 
            roads: a list of tuples, (u, v, w) in integer
                directed edge from u to v, w is the weight, amount of time to take to pass
            solulus: list of tuples, (a, b,c) in integer
                a is the id of solulu tree
                b is the amount of time to destroy
                c is which tree to teleport to
        """
        self.roads = roads
        self.solulus = solulus
        self.num_vertices = self.get_num_uniq_trees(roads) #O(R)

        # Add an array of vertex objects that the graph has
        self.vertices = [None] * self.num_vertices #O(T)
        for i in range(self.num_vertices): #O(T)
            self.vertices[i] = Vertex(i) 
        #Add edges of the graph from input
        self.add_edges(roads) #O(R)


    def get_num_uniq_trees(self, roads):
        
        """
        Function description: 
            Get the number of unique vertices(uniq trees) from input of edge information. 
            ID is assigned sequentially from 0 to ... , so just get the max num of vertex and +1 for getting uniq num of vertex
        Time complexity: O(R) where R is the number of roads
        Aux Space complexity: O(1)
        """
        max_num = 0
        for i in range(len(roads)):
            if roads[i][0] > max_num:
                max_num = roads[i][0]
            if roads[i][1] > max_num:
                max_num = roads[i][1]
        return max_num + 1 # +1 for id 0

    def reset(self):
        """
        Function description: 
            Reset the vertex info and make sure escape() can run again and again
        Time complexity: O(T) where T is the set of unique trees.
        Aux Space complexity: O(1)
        """
        for vertex in self.vertices:
            vertex.discovered = False
            vertex.visited = True
            vertex.time = float("inf")
            vertex.previous = None
            self.previous_time = -1 # store the time(weight) b/w u to v road

    def add_edges(self, edges_list): #edge_info:  [(u, v, w)...]
        """ 
        Function description: 
            Add directed edges in the graph from input, set of edges.
        Time complexity: O(R) where R is the number of roads
        Aux Space complexity: O(1)
        """
        for edge in edges_list:
            u = edge[0]
            v = edge[1]
            w = edge[2]
            edge_obj = Edge(u,v,w)  # make edge obj and vertex u will have that edge 
            vertex_obj = self.vertices[u]
            vertex_obj.add_edge(edge_obj)

    def __str__(self):
        """
        Function description: 
            Output the id of all vertices in the graph
        Time complexity: O(T) where T is the set of unique trees.
        Aux Space complexity: O(1)
        """
        return_string = " "
        for vertex in self.vertices:
            return_string +=  str(vertex) + "\n" # call the str method in Vertex class
        return return_string

    def escape(self, start, exits):
        """
        Function description:
            This function return the fastest route in id from start tree to one of exits trees and amount of time that takes.
            If no such route exist, return None.
        Approach description:
            Create duplicated(the same) graph of original graph and places exits only in 2nd(duplicated) graph. 
            Original graph is the pre-destroy-solulu graph and 2nd one is the post-destroy-solulu one.
            Moreover, create and connect the  edges that connects pre-graph and post-graph by connecting solulu tree in the original graph to a tree of teleportation destination(teleportation) or the same solulu tree(no teleportation) in the 2nd graph.
            Weights of edge are the time to take to destroy the solulu tree. Now, you can only reach exits after passing one of those edges(roads) (passing the road means destroying solulu tree also), which means after destroying one solulu trees.
            For example, if Tree 0 is supposed to teleport to Tree 2,create and connect the road(edge) of tree 0 in original graph to Tree 2 in the 2nd graph.
            Even if some solulu trees has no teleportation role, for example, tree 1 is solulu tree but not allow teleportation. Create and connect the road(edge) of tree 1 in original graph and Tree 1 in the 2nd graph.
            However, I cannot use the same id of the original graph for the 2nd graph, so I use id from max id of original graph +1.
            For example, the original graph has 0,1,2,3,4 id of vertices, 2nd duplicated graph will have 5,6,7,8,9.(vertex 0 == vertex 5, v1 == v6...and so on)
            By adding number of vertices, 5 to each id of original graph, I can get id of the 2nd graph.
            Therefore, to get input data, road for 2nd graph, I just add number of vertices to each values of all self.roads.(line 370)
            To get a combined graph of both original and 2nd graph, I call the constructor with input of both original and 2nd graph.
            Then I connects edges from original to 2nd graph, and run dijkstra and find fastest route.
            [How to find one of the exits(optimal)]
            Create the dummy vertex and connect with all exit trees with 0 weighted edges. The moment I reach this dummy vertex, that means I have passed one of exits.
            Therefore if I go back one vertex before from dummy vertex, I can find optimal exit.
        Input:
            start: non-negative integer id of source vertex(tree)
            exits: non-empty list of non-negative integer id that represent exit trees.
        Output: (total time, route), tuple, total time is represented as integer and route is represented as a list of integers(id)
        Time complexity: O(R log T) where R is the set of roads and T is the set of unique trees
            Time complexity analysis:
                1.)[duplicated graph part] O(T+R) where R is the set of roads and T is the set of unique trees
                because some for loop iterate T times and calling a constructor takes O(T+R) and so on...but to sum up, overall is O(T+R)
                However, O(T+R) is bounded by O(R).
                2.)[Dijkstra part] O(T^2log T) where R is the set of roads and T is the set of unique trees
                because in the worst case, while iterate T times, "for edge in u.edges" iterate T times, "min_heap.add((v, v.time))" takes O(logV).
                Therefore, overall time complexity of Dijkstra(not escape function) takes 0(T* T* log T) = O(T^2log T)
                In the worst case, graph is dense, so I can say that O(T^2log T) = O(R logT) because R = T^2 when graph is dense.
                3.) [backtracking] backtracking takes O(T) where T is the set of unique trees
                [Overall: time complexity of escape()]
                Time complexity of escape() overall is 1.) + 2.) + 3.) = O(R) + O(R logT) +O(T) = O(R+T) + O(R log T) = O(R) + O(R log T) = O(R log T)
        Space complexity: O(R+T) where R is the set of roads and T is the set of unique trees
            Input space analysis: O(1) because input arguments are the integer
            Aux space analysis: O(R+T): I need O(R) Aux space for storing new road(the 2nd graph) and O(T) Aux space for new id of solulu and new self.vertices for new graph, and heap.
        """
        self.reset() #O(T)
        
        # =====================create [duplicated graph] part =============================================
        
        #create the same graph and connects by edges describing roads of solulu trees and teleportation
        new_roads = []
        # get the number of uniq vertices(original graph) before adding 2nd graph
        self.initial_num_v = self.num_vertices
        #getting roads for 2nd graph adding the number of q¥uniq vertices in the original graph
        for tuple in self.roads: #O(R)
            new_road = (tuple[0] + self.initial_num_v, tuple[1]+ self.initial_num_v, tuple[2])
            new_roads.append(new_road)
        new_exits = []
        #change the id of exits to the new one(place exits only in the 2nd graph)
        for i in exits: #O(T)
            new_exits.append(i+ self.initial_num_v)
        # set id of dummy vertex
        id_dummy_var = self.initial_num_v * 2
        # connect all exits with dummy vertex
        for i in new_exits: #O(T)
            new_roads.append((i, id_dummy_var, 0))
        
        roads = self.roads + new_roads # max(O(T)+ O(R)) ->O(R)
        self.__init__(roads,self.solulus) #O(T+R)
        #Connects the original graph and the 2nd graph
        for solulu in self.solulus: #O(T)
            id = solulu[0]
            self.vertices[id].add_edge(Edge(solulu[0],solulu[2]+ self.initial_num_v, solulu[1])) 

        # =======================================[Dijkstra] start =================================================
        start = self.vertices[start]
        start.time = 0
        
        # min heap will have the 2 times sized array than it supposed to be to handle 2 graphs in escape() in the future 
        min_heap = MinHeap(self.initial_num_v * 2) #O(T)
        min_heap.add((start, start.time)) #O(log T)
        start.discovered = True
        while len(min_heap) > 0 : # until I visit all vertices, keep traversing
            u = min_heap.serve() # u : current vertex I am at
            u.visited = True # I have visited u and time is finalised 
            
            if u.id == id_dummy_var: # When I reach dummy vertex, which means I have arrived(and passed) at one exit before
                des = u.previous.id
            
            for edge in u.edges: #load all adjacent(connected directed) edges of u 
                v = edge.v # one of adjacent vertex of u
                v = self.vertices[v]
                if v.discovered == False: # first time of visit, so put initial info (vertex, time) to heap (time is not finalised)
                    v.time =  u.time + edge.w
                    min_heap.add((v, v.time)) #O(log T)
                    v.previous = u
                    v.previous_time = edge.w
                    v.discovered = True
                #edge relaxation
                elif v.visited == False and v.discovered == True: #Even if I have ever discovered that vertex before, can update to the faster time 
                    if v.time > u.time + edge.w: # choose faster time one and update time
                        v.time = u.time + edge.w
                        v.previous = u #update previous node as well
                        v.previous_time = edge.w
                        min_heap.update(v, v.time)# update elem/ item in the heap
        
        start = start.id
        # using start(source) and optimal destination(fastest exit) I found using dummy vertex, I backtrack and compute the vertices and time taken
        try:
            return self.backtracking(start, des)
        except: # if no path(roads) taken, return None
            return None
    
    def backtracking(self,start, end):
        """
        Function description:
            From start(source) and destination information, calculate the amount of time taken and shortest path.
        Approach description:
            In Dijkstra alg, each vertex has info of previous vertex(which vertex I was from) and amount of time taken.
            Therefore, I can track the path from destination to start vertex. For better complexity, I append the path reversely(destination to start) first by using append(),
            and them I reverse the entire path using reverse().
            Since 2nd graph has different ids than original graph, I changed each of those ids to the correct ids and added to the path.
            To remove id duplication of ids(solulu in the original graph to solulu in the 2nd graph) I set additional condition to avoid that.
        Time complexity: O(T) where T is the set of unique trees
            Time complexity analysis: 
                while iterate 2T times since it traverse 2T number of vertices (original+ 2nd graph) and reverse() takes O(T)
                while block takes O(1), Therefore, overall O(T)
        Aux Space complexity: O(T) where T is the set of unique trees
            Input space analysis: O(1) since input is the integer
            Aux space analysis:path will take O(T) aux space because it holds all vertices in the graph in the worst case"""
        time = 0
        path = []
        start = self.vertices[start]
        end = self.vertices[end]
        traverse_tree = end
        if end.id >= self.initial_num_v:
            end.id -= self.initial_num_v
        path.append(end.id)
        time += end.previous_time
        while traverse_tree is not start: # if path has more than 2 vertices
            previous = traverse_tree.previous
            if previous.id >= self.initial_num_v: # convert id of 2nd graph to the one of original graph
                previous.id -= self.initial_num_v
            if not traverse_tree.id == previous.id:# avoid storing the same id(id of 2nd graph and original graph might duplicate when teleportation)
                    path.append(previous.id)
            time += previous.previous_time
            traverse_tree =  previous
            
        path.reverse() # O(T)
        return (time, path)


class Vertex:
    def __init__(self, id) -> None:
        """
        Function description: Constructor of Vertex class.
        Input: id of an integer 
		Time complexity: O(1)
        Auxiliary space complexity: O(1)
        """
        
        # Basic info of Vertex
        self.id = id # name of vertex(number)
        self.edges = [] # a list of edges connected with the Vertex
        self.discovered = False
        self.visited = False
        #time: how long the Vertex takes from source
        self.time = float("inf") # for dijkstra, initialize by inf
        #for backtracking: where I was from
        self.previous = None
        self.previous_time = 0 # store the time (weight) b/w u to v


    def add_edge(self, edge):
        """
        Function description: Add edges to particular vertex
        Time complexity: O(1)
        Auxiliary space complexity: O(1)
        """
        self.edges.append(edge)


    def __str__(self):
        """
        Function description: output outgoing edges of particular vertex
        Time complexity: O(n) where n is the number of outgoing edges or adjacent vertices
        Auxiliary space complexity: O(1)
        """
        return_string = "Vertex " + str(self.id) 
        for edge in self.edges:
            return_string = return_string +  "\n" +" is connected with " + str(edge) 
        return return_string


class Edge:
    def __init__(self, u, v, w) -> None:
        """
        Function description: Constructor of edges
        Edge is defined from two vertices (and weight)
        Time complexity: O(1)
        Auxiliary space complexity: O(1)
        """
        self.u = u # current vertex
        self.v = v #adjacent vertex
        self.w = w

    def __str__(self):
        """
        Function description: Output edge info (u,v,w)
        Time complexity: O(1)
        Auxiliary space complexity: O(1)
        """
        return_string = "(" + str(self.u) + ", "+ str(self.v) + ","+ str(self.w) + ")"
        return return_string


if __name__ == "__main__":
    forest_test = TreeMap([(0,1,1), (1,2,1), (2,3,1), (0,2,1), (1,3,1), (3,0,1)], [(0,2,3), (1,1,2), (3,1,0), (2,3,0)])
    assert (forest_test.escape(0, [2]) == (3, [0, 1, 3, 0]) ) # error
    assert (forest_test.escape(0, [0]) == (3, [0, 1, 3, 0]) or forest_test.escape(0, [0]) == (3, [0, 3, 0]))
    assert (forest_test.escape(0, [0,1,2,3]) == (2, [0, 1, 2]) or forest_test.escape(0, [0,1,2,3]) == (2, [0, 3]))
    assert (forest_test.escape(3, [3]) == (3, [3, 0, 3]))
    assert (forest_test.escape(3, [0]) == (1, [3, 0]))
    assert (forest_test.escape(2, [0]) == (2, [2, 3, 0]))
    assert (forest_test.escape(1, [0]) == (2, [1, 3, 0]))
    assert (forest_test.escape(1, [3]) == (2, [1, 2, 3]))
    assert (forest_test.escape(1, [1]) == (3, [1, 3, 0, 1]))
    assert (forest_test.escape(2, [2]) == (3, [2, 3, 0, 2]))
    assert (forest_test.escape(4, [3]) == None)


