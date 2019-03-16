#ifndef _SNAPSHOT_H_
#define _SNAPSHOT_H_

#include "partition_hash.h"
#include "sbf.h"

/***
	Class for storing the snapshot of DPBF. Must contain the following functions - 

	Creation and Deletion :
		-	init_snapshot(b_node *a)
		-	free_snapshot(b_node *a)

	Helper functions :

		-	is_leaf(b_node *a)
		-	is_in_snapshot(UINT val, b_node *root)
		-	count_leaves(b_node *root)
		-	count_non_leaves(b_node *root)
		-	set_dirty_bit(UINT val, b_node *root)


	We do not require insertion function since we are not directly inserting in the snapshot.
	Instead we are updating the snapshot when required. The update functions are present in dpbf.h file.

***/

typedef struct b_node b_node;

// Node for the tree representing the snapshot
struct b_node {
	sbf *bf;
	b_node *left;
	b_node *right;
	int dirty_bit;
};

/**
	Initialzes the snapshot (whch is basically a tree of b_nodes)
	@param a : The root of the snapshot which is initialized
**/
void init_snapshot(b_node *a) {
	a->bf = NULL;
	a->left = NULL;
	a->right = NULL;
	a->dirty_bit = 0;
}

/**
	Frees a snapshot by recursively freeing it's children first
	@param curr_node : The curr_node in the recursion. It's children are freed first and then the node itself	
**/
void free_snapshot(b_node *curr_node) {
	if(curr_node == NULL) {
		return;
	} else {

		// Free it's children
		free_snapshot(curr_node->left);
		free_snapshot(curr_node->right);
	
		// Free the SBF
		if(curr_node->bf != NULL) {
			free_sbf(curr_node->bf);
			free(curr_node->bf);
		}	

		// Free the node itself
		free(curr_node);
	}
}

/**
	Checks if a b_node is a leaf or not
	@param node 	: The node which is to be checked
	@return bool 	: 1 is it's a leaf, 0 otherwise 
**/
int is_leaf(b_node *node) {
	if(node->left == NULL && node->right == NULL) {
		return 1;			// is leaf
	} else {
		return 0;			// is not leaf
	}
}

/**
	Checks if an element is already in the snapshot or not
	@param val 		: The value which is to be checked
	@param bloomf 	: The snapshot in which the value is to be checked 
**/
UINT is_in_snapshot(UINT val, b_node* bloomf) {
	b_node *curr_node = bloomf;
	UINT curr_level = 0;
	while(true) {
		if (curr_node->bf != NULL) {
			if(is_in_sbf(val, curr_node->bf)) {
				return 1;
			} else {
				return 0;							//element not present in the overall bloom filter
			}
		} else {
			if(partition_hash(val, curr_level) == 0) {
				curr_node = curr_node->left;
			} else {
				curr_node = curr_node->right;
			}
		}

		curr_level++;
	}
}

/**
	Counts the number of leaves in the snapshot
	@param node 	: The root of the snapshot
	@return count 	: The number of leaves 
**/
UINT count_leaves (b_node *node) {
	if(is_leaf(node)) {
		return 1;
	} else if (node->left == NULL) {
		printf("Caution:: Binary tree is not a full binary tree\n");
		return count_leaves(node->right);
	} else if (node->right == NULL) {
		printf("Caution:: Binary tree is not a full binary tree\n");
		return count_leaves(node->left);
	} else {
		return count_leaves(node->left) + count_leaves(node->right);
	}
}

/**
	Counts the number of non-leaves in the snapshot
	@param node 	: The root of the snapshot
	@return count 	: The number of non-leaves 
**/
UINT count_non_leaves(b_node *node) {
	if(is_leaf(node)) {
		return 0;
	} else if (node->left == NULL) {
		printf("Caution:: Binary tree is not a full binary tree\n");
		return 1 + count_non_leaves(node->right);
	} else if (node->right==NULL) {
		printf("Caution:: Binary tree is not a full binary tree\n");
		return 1 + count_non_leaves(node->left);
	} else {
		return 1 + count_non_leaves(node->left) + count_non_leaves(node->right);
	}
}

/**
	Sets the dirty-bit in the corresponding leaf of the snapshot
	@param val 		: The value which is currently being inserted in DPBF for which the dirty-bit is set
	@param root 	: The root of the snapshot
**/
void set_dirty_bit(UINT val, b_node *root) {
	b_node *curr_node = root;
	int curr_level = 0;
	int direction;

	while(!is_leaf(curr_node)) {

		direction = partition_hash(val, curr_level);

		if(direction == 0) {
			// Move left
			curr_node = curr_node->left;
		} else if(direction == 1) {
			// Move right
			curr_node = curr_node->right;
		} else {
			// Error
			printf("Partition functions exhausted (while setting dirty bit in snapshot)!\n\n\n");
			exit(0);
		}

		curr_level++;
	}

	// We've reached a leaf node, so set the dirty bit to 1
	curr_node->dirty_bit = 1;

	/*** 
		What if only one child of a node is NULL?
		It would not count as leaf so the above method will proceed. However, if we move 
		in the direction of the NULL child, the program will give wrong result/crash.

		Technically such a tree should not exist from our algo.
	***/	
}

#endif