#include "graph.cuh"
#include "gpu_error_check.cuh"
#include <cstdint>

template <class E>
Graph<E>::Graph(string graphFilePath, bool isWeighted)
{
	this->graphFilePath = graphFilePath;
	this->isWeighted = isWeighted;
}

template <class E>
string Graph<E>::GetFileExtension(string fileName)
{
    if(fileName.find_last_of(".") != string::npos)
        return fileName.substr(fileName.find_last_of(".")+1);
    return "";
}

template <>
void Graph<OutEdgeWeighted>::AssignW8(uint w8, uint index)
{
    edgeList[index].w8 = w8;
}

template <>
void Graph<OutEdge>::AssignW8(uint w8, uint index)
{
    edgeList[index].end = edgeList[index].end; // do nothing
}

template <class E>
void Graph<E>::ReadGraph()
{

	cout << "Reading the input graph from the following file:\n>> " << graphFilePath << endl;
	
	this->graphFormat = GetFileExtension(graphFilePath);
	
	if(graphFormat == "bcsr" || graphFormat == "bwcsr")
	{
		ifstream infile (graphFilePath, ios::in | ios::binary);
	
		infile.read ((char*)&num_nodes, sizeof(uint));
		infile.read ((char*)&num_edges, sizeof(uint));
		
		nodePointer = new uint[num_nodes+1];
		gpuErrorcheck(cudaMallocHost(&edgeList, (num_edges) * sizeof(E)));
		
		infile.read ((char*)nodePointer, sizeof(uint)*num_nodes);
		infile.read ((char*)edgeList, sizeof(E)*num_edges);
		nodePointer[num_nodes] = num_edges;
	}
	else if(graphFormat == "el" || graphFormat == "wel")
	{
		ifstream infile;
		infile.open(graphFilePath);
		stringstream ss;
		uint max = 0;
		string line;
		uint edgeCounter = 0;
		if(isWeighted)
		{
			vector<EdgeWeighted> edges;
			EdgeWeighted newEdge;
			while(getline( infile, line ))
			{
				ss.str("");
				ss.clear();
				ss << line;
				
				ss >> newEdge.source;
				ss >> newEdge.end;
				ss >> newEdge.w8;
				
				edges.push_back(newEdge);
				edgeCounter++;
				
				if(max < newEdge.source)
					max = newEdge.source;
				if(max < newEdge.end)
					max = newEdge.end;				
			}
			infile.close();
			num_nodes = max + 1;
			num_edges = edgeCounter;
			nodePointer = new uint[num_nodes+1];
			gpuErrorcheck(cudaMallocHost(&edgeList, (num_edges) * sizeof(E)));
			uint *degree = new uint[num_nodes];
			for(uint i=0; i<num_nodes; i++)
				degree[i] = 0;
			for(uint i=0; i<num_edges; i++)
				degree[edges[i].source]++;
			
			uint counter=0;
			for(uint i=0; i<num_nodes; i++)
			{
				nodePointer[i] = counter;
				counter = counter + degree[i];
			}
			nodePointer[num_nodes] = num_edges;
			uint *outDegreeCounter  = new uint[num_nodes];
			uint location;  
			for(uint i=0; i<num_edges; i++)
			{
				location = nodePointer[edges[i].source] + outDegreeCounter[edges[i].source];
				edgeList[location].end = edges[i].end;
				if(isWeighted)
					AssignW8(edges[i].w8, location);
					//edgeList[location].w8 = edges[i].w8;
				outDegreeCounter[edges[i].source]++;  
			}
			edges.clear();
			delete[] degree;
			delete[] outDegreeCounter;
			
		}
		else
		{
			vector<Edge> edges;
			Edge newEdge;
			while(getline( infile, line ))
			{
				ss.str("");
				ss.clear();
				ss << line;
				
				ss >> newEdge.source;
				ss >> newEdge.end;
				
				edges.push_back(newEdge);
				edgeCounter++;
				
				if(max < newEdge.source)
					max = newEdge.source;
				if(max < newEdge.end)
					max = newEdge.end;				
			}
			infile.close();
			num_nodes = max + 1;
			num_edges = edgeCounter;
			nodePointer = new uint[num_nodes+1];
			gpuErrorcheck(cudaMallocHost(&edgeList, (num_edges) * sizeof(E)));
			uint *degree = new uint[num_nodes];
			for(uint i=0; i<num_nodes; i++)
				degree[i] = 0;
			for(uint i=0; i<num_edges; i++)
				degree[edges[i].source]++;
			
			uint counter=0;
			for(uint i=0; i<num_nodes; i++)
			{
				nodePointer[i] = counter;
				counter = counter + degree[i];
			}
			nodePointer[num_nodes] = num_edges;
			uint *outDegreeCounter  = new uint[num_nodes];
			uint location;  
			for(uint i=0; i<num_edges; i++)
			{
				location = nodePointer[edges[i].source] + outDegreeCounter[edges[i].source];
				edgeList[location].end = edges[i].end;
				//if(isWeighted)
				//	edgeList[location].w8 = edges[i].w8;
				outDegreeCounter[edges[i].source]++;  
			}
			edges.clear();
			delete[] degree;
			delete[] outDegreeCounter;
		}
	}
	else if(graphFormat == "gr")
	{
		// GGR (Galois GR) binary format support
		// Layout: 32B header (4*uint64) + V*int64 row_start[1..V] + E*int32 edge_dst
		//         [+ 4B padding if weighted & E odd] + E*int32 adjwgt
		ifstream infile(graphFilePath, ios::in | ios::binary);

		uint64_t version, sizeEdgeTy, nvtxs64, nedges64;
		infile.read((char*)&version, sizeof(uint64_t));
		infile.read((char*)&sizeEdgeTy, sizeof(uint64_t));
		infile.read((char*)&nvtxs64, sizeof(uint64_t));
		infile.read((char*)&nedges64, sizeof(uint64_t));

		if (version != 1)
		{
			cout << "Unsupported GGR version: " << version << endl;
			exit(-1);
		}
		if (nvtxs64 > UINT_MAX || nedges64 > UINT_MAX)
		{
			cout << "Graph too large for Subway uint32: nvtxs=" << nvtxs64 << " nedges=" << nedges64 << endl;
			exit(-1);
		}

		num_nodes = (uint)nvtxs64;
		num_edges = (uint)nedges64;
		bool grWeighted = (sizeEdgeTy != 0);

		// Verify template parameter matches file content
		if (grWeighted != isWeighted)
		{
			cout << "GGR weight flag (" << grWeighted << ") does not match template parameter isWeighted (" << isWeighted << ")\n";
			exit(-1);
		}

		nodePointer = new uint[num_nodes + 1];
		gpuErrorcheck(cudaMallocHost(&edgeList, num_edges * sizeof(E)));

		// Read row_start[1..V] (int64 LE), convert to uint32 nodePointer[V]
		// GGR omits row_start[0]=0, Subway omits nodePointer[V] (sentinel added below)
		int64_t* rowStartBuf = new int64_t[num_nodes];
		infile.read((char*)rowStartBuf, sizeof(int64_t) * num_nodes);
		nodePointer[0] = 0;  // GGR-implicit row_start[0]
		for (uint i = 0; i < num_nodes; i++)
		{
			if (rowStartBuf[i] < 0 || (uint64_t)rowStartBuf[i] > UINT_MAX)
			{
				cout << "Row pointer overflow at vertex " << i << ": " << rowStartBuf[i] << endl;
				exit(-1);
			}
			nodePointer[i] = (uint)rowStartBuf[i];
		}
		nodePointer[num_nodes] = num_edges;
		delete[] rowStartBuf;

		// Read edges
		if (grWeighted)
		{
			// GGR weighted: separate dst[] + wgt[] arrays → interleave into OutEdgeWeighted
			int32_t* dstBuf = new int32_t[num_edges];
			infile.read((char*)dstBuf, sizeof(int32_t) * num_edges);

			// Skip 4-byte padding if edge count is odd (GGR aligns adjwgt to 8 bytes)
			if (num_edges % 2 == 1)
				infile.seekg(4, ios::cur);

			int32_t* wgtBuf = new int32_t[num_edges];
			infile.read((char*)wgtBuf, sizeof(int32_t) * num_edges);

			for (uint i = 0; i < num_edges; i++)
			{
				edgeList[i].end = (uint)dstBuf[i];
				AssignW8((uint)wgtBuf[i], i);
			}

			delete[] dstBuf;
			delete[] wgtBuf;
		}
		else
		{
			// GGR unweighted: edge_dst is int32 LE, sizeof(OutEdge)=4=sizeof(int32)
			// Direct read safe on LE: valid vertex IDs [0, V-1] are positive int32
			infile.read((char*)edgeList, sizeof(E) * num_edges);
		}
	}
	else
	{
		cout << "The graph format is not supported!\n";
		exit(-1);
	}

	outDegree  = new unsigned int[num_nodes];

	for(uint i=1; i<num_nodes-1; i++)
		outDegree[i-1] = nodePointer[i] - nodePointer[i-1];
	outDegree[num_nodes-1] = num_edges - nodePointer[num_nodes-1];

	label1 = new bool[num_nodes];
	label2 = new bool[num_nodes];
	value  = new unsigned int[num_nodes];

	gpuErrorcheck(cudaMalloc(&d_outDegree, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_value, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_label1, num_nodes * sizeof(bool)));
	gpuErrorcheck(cudaMalloc(&d_label2, num_nodes * sizeof(bool)));

	cout << "Done reading.\n";
	cout << "Number of nodes = " << num_nodes << endl;
	cout << "Number of edges = " << num_edges << endl;


}

//--------------------------------------

template <class E>
GraphPR<E>::GraphPR(string graphFilePath, bool isWeighted)
{
	this->graphFilePath = graphFilePath;
	this->isWeighted = isWeighted;
}

template <class E>
string GraphPR<E>::GetFileExtension(string fileName)
{
    if(fileName.find_last_of(".") != string::npos)
        return fileName.substr(fileName.find_last_of(".")+1);
    return "";
}

template <>
void GraphPR<OutEdgeWeighted>::AssignW8(uint w8, uint index)
{
    edgeList[index].w8 = w8;
}

template <>
void GraphPR<OutEdge>::AssignW8(uint w8, uint index)
{
    edgeList[index].end = edgeList[index].end; // do nothing
}

template <class E>
void GraphPR<E>::ReadGraph()
{

	cout << "Reading the input graph from the following file:\n>> " << graphFilePath << endl;
	
	this->graphFormat = GetFileExtension(graphFilePath);
	
	if(graphFormat == "bcsr" || graphFormat == "bwcsr")
	{
		ifstream infile (graphFilePath, ios::in | ios::binary);
	
		infile.read ((char*)&num_nodes, sizeof(uint));
		infile.read ((char*)&num_edges, sizeof(uint));
		
		nodePointer = new uint[num_nodes+1];
		gpuErrorcheck(cudaMallocHost(&edgeList, (num_edges) * sizeof(E)));
		
		infile.read ((char*)nodePointer, sizeof(uint)*num_nodes);
		infile.read ((char*)edgeList, sizeof(E)*num_edges);
		nodePointer[num_nodes] = num_edges;
	}
	else if(graphFormat == "el" || graphFormat == "wel")
	{
		ifstream infile;
		infile.open(graphFilePath);
		stringstream ss;
		uint max = 0;
		string line;
		uint edgeCounter = 0;
		if(isWeighted)
		{
			vector<EdgeWeighted> edges;
			EdgeWeighted newEdge;
			while(getline( infile, line ))
			{
				ss.str("");
				ss.clear();
				ss << line;
				
				ss >> newEdge.source;
				ss >> newEdge.end;
				ss >> newEdge.w8;
				
				edges.push_back(newEdge);
				edgeCounter++;
				
				if(max < newEdge.source)
					max = newEdge.source;
				if(max < newEdge.end)
					max = newEdge.end;				
			}
			infile.close();
			num_nodes = max + 1;
			num_edges = edgeCounter;
			nodePointer = new uint[num_nodes+1];
			gpuErrorcheck(cudaMallocHost(&edgeList, (num_edges) * sizeof(E)));
			uint *degree = new uint[num_nodes];
			for(uint i=0; i<num_nodes; i++)
				degree[i] = 0;
			for(uint i=0; i<num_edges; i++)
				degree[edges[i].source]++;
			
			uint counter=0;
			for(uint i=0; i<num_nodes; i++)
			{
				nodePointer[i] = counter;
				counter = counter + degree[i];
			}
			nodePointer[num_nodes] = num_edges;
			uint *outDegreeCounter  = new uint[num_nodes];
			uint location;  
			for(uint i=0; i<num_edges; i++)
			{
				location = nodePointer[edges[i].source] + outDegreeCounter[edges[i].source];
				edgeList[location].end = edges[i].end;
				if(isWeighted)
					AssignW8(edges[i].w8, location);
					//edgeList[location].w8 = edges[i].w8;
				outDegreeCounter[edges[i].source]++;  
			}
			edges.clear();
			delete[] degree;
			delete[] outDegreeCounter;
			
		}
		else
		{
			vector<Edge> edges;
			Edge newEdge;
			while(getline( infile, line ))
			{
				ss.str("");
				ss.clear();
				ss << line;
				
				ss >> newEdge.source;
				ss >> newEdge.end;
				
				edges.push_back(newEdge);
				edgeCounter++;
				
				if(max < newEdge.source)
					max = newEdge.source;
				if(max < newEdge.end)
					max = newEdge.end;				
			}
			infile.close();
			num_nodes = max + 1;
			num_edges = edgeCounter;
			nodePointer = new uint[num_nodes+1];
			gpuErrorcheck(cudaMallocHost(&edgeList, (num_edges) * sizeof(E)));
			uint *degree = new uint[num_nodes];
			for(uint i=0; i<num_nodes; i++)
				degree[i] = 0;
			for(uint i=0; i<num_edges; i++)
				degree[edges[i].source]++;
			
			uint counter=0;
			for(uint i=0; i<num_nodes; i++)
			{
				nodePointer[i] = counter;
				counter = counter + degree[i];
			}
			nodePointer[num_nodes] = num_edges;
			uint *outDegreeCounter  = new uint[num_nodes];
			uint location;  
			for(uint i=0; i<num_edges; i++)
			{
				location = nodePointer[edges[i].source] + outDegreeCounter[edges[i].source];
				edgeList[location].end = edges[i].end;
				//if(isWeighted)
				//	edgeList[location].w8 = edges[i].w8;
				outDegreeCounter[edges[i].source]++;  
			}
			edges.clear();
			delete[] degree;
			delete[] outDegreeCounter;
		}
	}
	else if(graphFormat == "gr")
	{
		// GGR (Galois GR) binary format support
		// Layout: 32B header (4*uint64) + V*int64 row_start[1..V] + E*int32 edge_dst
		//         [+ 4B padding if weighted & E odd] + E*int32 adjwgt
		ifstream infile(graphFilePath, ios::in | ios::binary);

		uint64_t version, sizeEdgeTy, nvtxs64, nedges64;
		infile.read((char*)&version, sizeof(uint64_t));
		infile.read((char*)&sizeEdgeTy, sizeof(uint64_t));
		infile.read((char*)&nvtxs64, sizeof(uint64_t));
		infile.read((char*)&nedges64, sizeof(uint64_t));

		if (version != 1)
		{
			cout << "Unsupported GGR version: " << version << endl;
			exit(-1);
		}
		if (nvtxs64 > UINT_MAX || nedges64 > UINT_MAX)
		{
			cout << "Graph too large for Subway uint32: nvtxs=" << nvtxs64 << " nedges=" << nedges64 << endl;
			exit(-1);
		}

		num_nodes = (uint)nvtxs64;
		num_edges = (uint)nedges64;
		bool grWeighted = (sizeEdgeTy != 0);

		if (grWeighted != isWeighted)
		{
			cout << "GGR weight flag (" << grWeighted << ") does not match template parameter isWeighted (" << isWeighted << ")\n";
			exit(-1);
		}

		nodePointer = new uint[num_nodes + 1];
		gpuErrorcheck(cudaMallocHost(&edgeList, num_edges * sizeof(E)));

		int64_t* rowStartBuf = new int64_t[num_nodes];
		infile.read((char*)rowStartBuf, sizeof(int64_t) * num_nodes);
		nodePointer[0] = 0;  // GGR-implicit row_start[0]
		for (uint i = 0; i < num_nodes; i++)
		{
			if (rowStartBuf[i] < 0 || (uint64_t)rowStartBuf[i] > UINT_MAX)
			{
				cout << "Row pointer overflow at vertex " << i << ": " << rowStartBuf[i] << endl;
				exit(-1);
			}
			nodePointer[i] = (uint)rowStartBuf[i];
		}
		nodePointer[num_nodes] = num_edges;
		delete[] rowStartBuf;

		if (grWeighted)
		{
			int32_t* dstBuf = new int32_t[num_edges];
			infile.read((char*)dstBuf, sizeof(int32_t) * num_edges);

			if (num_edges % 2 == 1)
				infile.seekg(4, ios::cur);

			int32_t* wgtBuf = new int32_t[num_edges];
			infile.read((char*)wgtBuf, sizeof(int32_t) * num_edges);

			for (uint i = 0; i < num_edges; i++)
			{
				edgeList[i].end = (uint)dstBuf[i];
				AssignW8((uint)wgtBuf[i], i);
			}

			delete[] dstBuf;
			delete[] wgtBuf;
		}
		else
		{
			// GGR unweighted: sizeof(OutEdge)=4=sizeof(int32), direct read safe on LE
			infile.read((char*)edgeList, sizeof(E) * num_edges);
		}
	}
	else
	{
		cout << "The graph format is not supported!\n";
		exit(-1);
	}

	outDegree  = new unsigned int[num_nodes];

	for(uint i=1; i<num_nodes-1; i++)
		outDegree[i-1] = nodePointer[i] - nodePointer[i-1];
	outDegree[num_nodes-1] = num_edges - nodePointer[num_nodes-1];


	value  = new float[num_nodes];
	delta  = new float[num_nodes];

	gpuErrorcheck(cudaMalloc(&d_outDegree, num_nodes * sizeof(unsigned int)));
	gpuErrorcheck(cudaMalloc(&d_value, num_nodes * sizeof(float)));
	gpuErrorcheck(cudaMalloc(&d_delta, num_nodes * sizeof(float)));


	cout << "Done reading.\n";
	cout << "Number of nodes = " << num_nodes << endl;
	cout << "Number of edges = " << num_edges << endl;


}


template class Graph<OutEdge>;
template class Graph<OutEdgeWeighted>;

template class GraphPR<OutEdge>;
template class GraphPR<OutEdgeWeighted>;
