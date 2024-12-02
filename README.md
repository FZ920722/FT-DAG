# FT-DAG

FT-DAG is an efficient and formally verified full-topology DAG generator that is able to control all major parameters, including:

- in-degree: _id; 
- out-degree: _od;
- width: _wd, _wu; 
- jump level: _jl; 
- hang level: _hl;
- length: _ld, _lu;
- shape value: _sd, _su;
- the number of nodes: _n;
- the number of edges: _md, _mu;

Experiments show that when the number of nodes is larger than 20, FT-DAG provides at least two orders of magnitude speedup compared to the state of the art and more orders to other generators. FT-DAG scales to 100 nodes in a typical industrial case study within hours.