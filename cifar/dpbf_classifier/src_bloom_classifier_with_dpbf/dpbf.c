#ifndef _DPBF_H_
#define _DPBF_H_

#include <iostream>
#include <map>
#include <list>
#include "snapshot.h"
#include "sbf.h"
#include "partition_hash.h"
#include "bloom_parameters.h"

using namespace std;

/***
	Class for DPBF. Consists of a snapshot and a pbf_list. Must have the following functions -

	Creation and Deletion :
		-	init_dpbf(dpbf *a)
		-	free dpbf(dpbf *a)

	Insertion and other helper funcitons : 
		-	insert_in_dpbf(UINT val, dpbf *a)
		-	update_snapshot()
		-	bottom_up(bf_linked_list *pbf_list, int start, int end)
		-	is_in_dpbf(UINT val, dpbf *a)

	Union and intersection : 
		- 	union_dpbf(dpbf *a, dpbf *b)
		-	intersection_dpbf(dpbf *a, dpbf *b)

***/

typedef struct dpbf dpbf;

struct dpbf {
	map<UINT, sbf*> *myMap;
	b_node *snapshot;
};

bloom_parameters *bloomParam;

/***
	Auxillary data structure for new construction of snapshot
***/

typedef struct list_node list_node;

struct list_node { 
	b_node *node;
	UINT index;
	list_node* next;
};


/**
	Initializes the DPBF
	@param a : The DPBF to be initialized
**/
void init_dpbf(dpbf *a) {
	a->snapshot = (b_node*) malloc(sizeof(b_node));
	init_snapshot(a->snapshot);
	a->myMap = new map<UINT, sbf*> ();
}

/**
	Initializes the DPBF without map
	@param a : The DPBF to be initialized
**/
void init_dpbf_without_map(dpbf *a) {
	a->snapshot = (b_node*) malloc(sizeof(b_node));
	init_snapshot(a->snapshot);
}

/**
	Frees a DPBF
	@param a : The DPBF which is to be freed
**/
void free_dpbf(dpbf *a) {
	free_snapshot(a->snapshot);
	free(a);
}

/**
	Inserts a value in DPBF
	@param val 	: The value to be inserted
	@param a 	: The DPBF in which the value is to be inserted
**/
void insert_in_dpbf(UINT val, dpbf *a) {
	UINT index = 0;

	// Find the index of the bloom filter in the linked list
	for(int curr_level = 0; curr_level < bloomParam->NUM_PARTITION; curr_level++) {
		index = index * 2 + partition_hash(val, curr_level);
	}

	map<UINT, sbf*>::iterator it = a->myMap->find(index);

	if(it != a->myMap->end()) {
		// Insert in sbf
		if(insert_in_sbf(val, it->second)) {
			set_dirty_bit(val, a->snapshot);
		}
	} else {
		// create a new node and insert in map
		sbf *new_sbf = (sbf *) malloc(sizeof(sbf));
		init_sbf(new_sbf);
		insert_in_sbf(val, new_sbf);

		a->myMap->insert(pair<UINT, sbf*>(index, new_sbf));
		set_dirty_bit(val, a->snapshot);
	}
}

/**
	Helper function used by update_snapshot(). It frees the linked list node (essentially deleting it) which we create 
	to store the nodes of pbf_list in the given range
	@param a : The node to be freed
**/
void free_list_node(list_node* a) {
	free(a);
}

/**
	Updates the snapshot using the pbf_list and the dirty bits set in the snapshot.
	@param root			: Root of the snapshot
	@param curr_value	: The index of the root node if the tree was stored in a heap
	@param curr_height 	: The height of the root node
	@param ll 			: The pbf_list
	@return return_node	: The root of the updated snapshot
**/
b_node* update_snapshot(b_node *root, UINT curr_value, UINT curr_height, map<UINT, sbf*> *myMap) {
	UINT start, end;
	start = curr_value * pow(2, bloomParam->NUM_PARTITION - curr_height);
	end = start + pow(2, bloomParam->NUM_PARTITION - curr_height) - 1;
	UINT bottom_up_height = bloomParam->NUM_PARTITION;

	// Free current root
	free_snapshot(root);

	list_node *start_node = NULL, *curr_node, *del_node;

	map<UINT, sbf*>::iterator it = myMap->lower_bound(start);

	UINT num_pbf_nodes = 0;
	while(it != myMap->end() && it->first <= end) {
		b_node *temp_node = (b_node*) malloc(sizeof(b_node));
		init_snapshot(temp_node);

		// Initialize the bloom filter of the b_node
		temp_node->bf = (sbf*)malloc(sizeof(sbf));
		init_sbf(temp_node->bf);

		// Copy the bloom filter of the full tree in the one for temp node
		copy_sbf(temp_node->bf, it->second);
		list_node* l_node = (list_node*)malloc(sizeof(list_node));
		l_node->node = temp_node;
		l_node->index = it->first;
		l_node->next = NULL;

		if(start_node == NULL) {
			start_node = l_node;
			curr_node = l_node;
			num_pbf_nodes++;
		} else {
			curr_node->next = l_node;
			curr_node = curr_node->next;
			num_pbf_nodes++;
		}

		it++;
	}

	if(num_pbf_nodes == 0) {
		
		root = (b_node*) malloc(sizeof(b_node));
		init_snapshot(root);

		// Initialize the bloom filter of the b_node
		root->bf = (sbf*)malloc(sizeof(sbf));
		init_sbf(root->bf);

	} else if(num_pbf_nodes == 1) {

		root = start_node->node;

	} else {
		
		UINT l_index;
		while(num_pbf_nodes > 1) {
			curr_node = start_node;
			bottom_up_height --;

			while(curr_node != NULL) {

				l_index = curr_node->index;
				if(l_index % 2 == 0) {
					// Even
					if(is_leaf(curr_node->node)) {
						// Leaf
						if(curr_node->next != NULL && (curr_node->next)->index == l_index + 1) {
							if (is_leaf((curr_node->next)->node) && ((((curr_node->node)->bf)->num_elems + (((curr_node->next)->node)->bf)->num_elems) < (bloomParam->FILL_THRESHOLD + 1))) {
								// Take union
								UINT total_elem = ((curr_node->node)->bf)->num_elems + (((curr_node->next)->node)->bf)->num_elems;
								union_sbf((curr_node->node)->bf, ((curr_node->next)->node)->bf, (curr_node->node)->bf);
								((curr_node->node)->bf)->num_elems = total_elem;
								del_node = curr_node->next;
								curr_node->next = (curr_node->next)->next;
								free_list_node(del_node);
								num_pbf_nodes--;
							} else {
								// Create parent node
								b_node *parent_node = (b_node*) malloc(sizeof(b_node));
								init_snapshot(parent_node);

								parent_node->left = curr_node->node;
								parent_node->right = (curr_node->next)->node;

								curr_node->node = parent_node;
								del_node = curr_node->next;
								curr_node->next = (curr_node->next)->next;
								free_list_node(del_node);
								num_pbf_nodes--;
							}
						} 
					} else { 
						// Not leaf
						if(curr_node->next != NULL && (curr_node->next)->index == l_index + 1) {
							// Create parent node
							b_node *parent_node = (b_node*) malloc(sizeof(b_node));
							init_snapshot(parent_node);

							parent_node->left = curr_node->node;
							parent_node->right = (curr_node->next)->node;

							curr_node->node = parent_node;
							del_node = curr_node->next;
							curr_node->next = (curr_node->next)->next;
							free_list_node(del_node);
							num_pbf_nodes--;
						
						} else {
							// Create parent node
							b_node *right_node = (b_node*) malloc(sizeof(b_node));
							init_snapshot(right_node);

							// Initialize the bloom filter of the right_node
							right_node->bf = (sbf*)malloc(sizeof(sbf));
							init_sbf(right_node->bf);

							// Create parent node
							b_node *parent_node = (b_node*) malloc(sizeof(b_node));
							init_snapshot(parent_node);

							parent_node->left = curr_node->node;
							parent_node->right = right_node;

							curr_node->node = parent_node;
					
						}
					}
				} else {
					// Odd
					if(!is_leaf(curr_node->node)) {
						// Not Leaf
						// Create parent node
						b_node *left_node = (b_node*) malloc(sizeof(b_node));
						init_snapshot(left_node);

						// Initialize the bloom filter of the right_node
						left_node->bf = (sbf*)malloc(sizeof(sbf));
						init_sbf(left_node->bf);

						// Create parent node
						b_node *parent_node = (b_node*) malloc(sizeof(b_node));
						init_snapshot(parent_node);

						parent_node->left = left_node;
						parent_node->right = curr_node->node;

						curr_node->node = parent_node;
					} 
				}	

				// Update index
				curr_node->index = l_index / 2;
				curr_node = curr_node->next;
			}
		}

		root = start_node->node;

		// In case only one node remains but we still haven't bottom-up'ed to the height of the ""root"" node.
		// This will only happen in the upper levels of the snapshot (assuming snapshot is small) or it'll happen 
		// in the worst case for dpbf in which all the elements are inserted in two consecutive nodes of pbf_list
		// In that case, our snapshot will degenerate into a list like structure having many leaf nodes with sbf's 
		// which do not have any element in them
		if(!is_leaf(root)) {
			while(bottom_up_height > curr_height) {
				bottom_up_height --;
				l_index = start_node->index;
				
				if(l_index % 2 == 0) {
					// Create parent node
					b_node *right_node = (b_node*) malloc(sizeof(b_node));
					init_snapshot(right_node);

					// Initialize the bloom filter of the right_node
					right_node->bf = (sbf*)malloc(sizeof(sbf));
					init_sbf(right_node->bf);

					// Create parent node
					b_node *parent_node = (b_node*) malloc(sizeof(b_node));
					init_snapshot(parent_node);

					parent_node->left = start_node->node;
					parent_node->right = right_node;

					start_node->node = parent_node;
				} else {
					// Create parent node
					b_node *left_node = (b_node*) malloc(sizeof(b_node));
					init_snapshot(left_node);

					// Initialize the bloom filter of the right_node
					left_node->bf = (sbf*)malloc(sizeof(sbf));
					init_sbf(left_node->bf);

					// Create parent node
					b_node *parent_node = (b_node*) malloc(sizeof(b_node));
					init_snapshot(parent_node);

					parent_node->left = left_node;
					parent_node->right = start_node->node;

					start_node->node = parent_node;
				}
			}

			root = start_node->node;			
		}

	}

	return root;
}
/**
	Checks if an element is present in the dpbf or not
	@param val 	: The element to be checked
	@param a 	: The DPBF in whicih the element is to be checked
	@return ret : 0 or 1 depending on whether the element was present or not
**/
UINT is_in_dpbf(UINT val, dpbf *a) {
	// 0 -> not present; 1 -> present

	b_node *curr_node = a->snapshot, *parent_node = NULL;
	UINT curr_level = 0, curr_value = 0, curr_direction = 0;

	// If the root itself has dirty bit set but the snapshot isn't updated
	// While loop below cannot detect this condition since root->bf is NULL and the loop only checks if dirty bit is set or not for 
	// leaf nodes only which have bf != NULL
	if(curr_node->dirty_bit == 1 && curr_node->bf == NULL) {
		a->snapshot = update_snapshot(a->snapshot, curr_value, curr_level, a->myMap);
	}

	// If dpbf is empty, then return 
	if(curr_node->bf == NULL && curr_node->left == NULL && curr_node->right == NULL) {
		printf("DPBF empty\n");
		return 0;
	}

	// Find the value in the snapshot
	while(true) {

		// If we reach a leaf node
		if (is_leaf(curr_node)) {
			// If dirty-bit is set, then update (lazy propagate)
			if(curr_node->dirty_bit == 1) {

				if(parent_node == NULL) {
					a->snapshot = update_snapshot(a->snapshot, curr_value, curr_level, a->myMap);
				} else {
					if(curr_direction == 0) {
						parent_node->left = update_snapshot(parent_node->left, curr_value, curr_level, a->myMap);
						curr_node = parent_node->left;
					} else {
						parent_node->right = update_snapshot(parent_node->right, curr_value, curr_level, a->myMap);
						curr_node = parent_node->right;
					}
				}

				// To start at the same node in the next iteration
				curr_level--;				

			} else {

				// Dirty bit is not set for leaf node, so check if the element is present in sbf of that leaf node
				if(is_in_sbf(val, curr_node->bf)) {
					return 1;
				} else {
					return 0;							//element not present in the overall bloom filter
				}
			}
		} else {
			// Continue search
			parent_node = curr_node;
			if(partition_hash(val, curr_level) == 0) {
				curr_node = curr_node->left;
				curr_value = curr_value * 2;
				curr_direction = 0;
			} else {
				curr_node = curr_node->right;
				curr_value = curr_value * 2 + 1;
				curr_direction = 1;
			}
		}

		curr_level++;
	}

}

/**
	Takes union of two dpbf's and returns the same
	@param a,b 	: The dpbf's whose union is to be taken
	@return c 	: The union of a and b
**/
dpbf *union_dpbf(dpbf *a, dpbf *b) {
	dpbf *c;
	c = (dpbf*) malloc(sizeof(dpbf));
	init_dpbf_without_map(c);

	c->myMap = new map<UINT, sbf*> (*(a->myMap));
	map<UINT, sbf*>::iterator it = b->myMap->begin(), temp_it;
	
	for(;it != b->myMap->end(); it++) {
		temp_it = c->myMap->find(it->first);
		if(temp_it != c->myMap->end()) {
			union_sbf(temp_it->second, it->second, temp_it->second);
		} else {
			c->myMap->insert(pair<UINT, sbf*>(it->first, it->second));
		}
	}

	// Update snapshot
	c->snapshot = update_snapshot(c->snapshot, 0, 0, c->myMap);

	return c;
}

/**
	Takes intersection of two dpbf's and returns the same
	@param a,b 	: The dpbf's whose intersection is to be taken
	@return c 	: The intersection of a and b
**/
dpbf *intersection_dpbf(dpbf *a, dpbf *b) {
	dpbf *c;
	c = (dpbf*) malloc(sizeof(dpbf));
	init_dpbf(c);

	if(a->myMap->size() >= b->myMap->size()) {
		map<UINT, sbf*>::iterator it = b->myMap->begin(), temp_it;	
		
		for(;it != b->myMap->end(); it++) {
			temp_it = a->myMap->find(it->first);
			if(temp_it != a->myMap->end()) {
				// Same index found in both a and b so insert in map
				sbf* new_sbf = (sbf *) malloc(sizeof(sbf));
				init_sbf(new_sbf);
				intersect_sbf(temp_it->second, it->second, new_sbf);

				// Insert in the pbf_list of c
    	        if(num_ones_in_sbf(new_sbf) > 0) {
   	        		c->myMap->insert(pair<UINT, sbf*>(it->first, new_sbf));
    	        }
			}
		}

	} else {
		map<UINT, sbf*>::iterator it = a->myMap->begin(), temp_it;	
		
		for(;it != a->myMap->end(); it++) {
			temp_it = b->myMap->find(it->first);
			if(temp_it != b->myMap->end()) {
				// Same index found in both a and b so insert in map
				sbf* new_sbf = (sbf *) malloc(sizeof(sbf));
				init_sbf(new_sbf);
				intersect_sbf(temp_it->second, it->second, new_sbf);

				// Insert in the pbf_list of c
    	        if(num_ones_in_sbf(new_sbf) > 0) {
   	        		c->myMap->insert(pair<UINT, sbf*>(it->first, new_sbf));
    	        }
			}
		}
	}

	// Update snapshot
	c->snapshot = update_snapshot(c->snapshot, 0, 0, c->myMap);

	return c;
}

dpbf *mydpbf;
void init_bloom_filter(UINT SIZE_BLOOM ,UINT K ,UINT COUNTER_CHUNK,UINT NUM_PARTITION,float FALSE_PSTV_PROB)
{
	init_bloomParameters(SIZE_BLOOM ,K ,COUNTER_CHUNK,NUM_PARTITION,FALSE_PSTV_PROB);
	seive_initial();
	mydpbf = (dpbf*) malloc(sizeof(dpbf));
	init_dpbf(mydpbf);
	// mydpbf->snapshot = (b_node*) malloc(sizeof(b_node));
	// init_snapshot(mydpbf->snapshot);
	// mydpbf->myMap = new map<UINT, sbf*> ();	
	// return mydpbf;
}

void insert(UINT id)
{
	insert_in_dpbf(id, mydpbf);
}

UINT check(UINT id)
{
	return is_in_dpbf(id, mydpbf);
}

void freedpbf()
{
	free_dpbf(mydpbf);
}

void update()
{
	mydpbf->snapshot = update_snapshot(mydpbf->snapshot, 0, 0, mydpbf->myMap);
}

UINT getMemory()
{

	UINT msize = ((mydpbf->myMap)->size())*(sizeof(UINT) + sizeof(sbf*));
	return msize + count_leaves(mydpbf->snapshot) * (sizeof(b_node) + (bloomParam->SIZE_BLOOM / 8)) + count_non_leaves(mydpbf->snapshot) * sizeof(b_node);
}

UINT getMapSize()
{
	return ((mydpbf->myMap)->size())*(sizeof(UINT) + sizeof(sbf*));
}

UINT getTreeSize()
{
	return count_leaves(mydpbf->snapshot) * (sizeof(b_node) + (bloomParam->SIZE_BLOOM / 8)) + count_non_leaves(mydpbf->snapshot) * sizeof(b_node);
}

// dpbf* getP(dpbf &x)
// {
// 	return &x;
// }

#endif