// #include <map>
// // #include "snapshot.h"

// // using namespace std;
// typedef struct dpbf dpbf;
// typedef struct list_node list_node;
// typedef struct b_node b_node;

// // Node for the tree representing the snapshot
// struct b_node {
// 	sbf *bf;
// 	b_node *left;
// 	b_node *right;
// 	int dirty_bit;
// };
// #include"dpbf.c"
// struct dpbf {
// 	std::map<UINT, sbf*> *myMap;
// 	b_node *snapshot;
// };

// struct list_node { 
// 	b_node *node;
// 	UINT index;
// 	list_node* next;
// };

// UINT is_in_dpbf(UINT val, dpbf *a);
// dpbf* init_bloom_filter(UINT SIZE_BLOOM ,UINT K ,UINT COUNTER_CHUNK,UINT NUM_PARTITION,float FALSE_PSTV_PROB);
// void insert_in_dpbf(UINT val, dpbf *a);
// void free_dpbf(dpbf *a);
typedef unsigned long long int UINT;

void init_bloom_filter(UINT SIZE_BLOOM ,UINT K ,UINT COUNTER_CHUNK,UINT NUM_PARTITION,float FALSE_PSTV_PROB);
void insert(UINT id);
UINT check(UINT id);
void freedpbf();
UINT getMemory();
UINT getMapSize();
UINT getTreeSize();
void update();
