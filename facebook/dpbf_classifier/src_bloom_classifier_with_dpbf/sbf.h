#ifndef _SBF_H_
#define _SBF_H_

#include "bloom_parameters.h"
#include "hash.h"
#include "string.h"
#define MAX(a,b) ((a) > (b) ? a : b)
#define MIN(a,b) ((a) < (b) ? a : b)

/***
	Class for a standard bloom filter (SBF). Must define the following functions - 

	Creation and Deletion : 	
		-	init_sbf(sbf *a)
		-	free_sbf(sbf *a)

	Insertion and other helper functions :
		-	is_in_sbf(UINT val, sbf *a)
		-	insert_in_sbf(UINT val, sbf *a)
		-	count_elems_in_sbf(sbf *a)
		-	copy_sbf(sbf *a, sbf *b)		// copies b into a (similar to a = b)
	
	Union and intersection :
		-	union_sbf(sbf *a, sbf *b, sbf *c);
		-	intersect_sbf(sbf *a, sbf *b, sbf *c);
***/

typedef struct sbf sbf;

struct sbf {
	UINT *flag_array;
	UINT num_elems;
};



/**
	Initializes the SBF. 
	@param bl : The SBF to be initialized	
**/
void init_sbf(sbf *bl) {
 	int size = (bloomParam->SIZE_BLOOM / bloomParam->NUM_BITS);
	bl->flag_array = (UINT *) malloc(sizeof(UINT) * size);
	memset(bl->flag_array, 0, sizeof(UINT) * size);
	bl->num_elems = 0;
}

/**
	Frees an SBF.
	@param bl : The SBF to be freed
**/
void free_sbf(sbf *bl) {
	free(bl->flag_array);
}

/**
	Checks whether a given element is already in the SBF or not.
	@param val 	: The value to be checked
	@param bl  	: The SBF in which the value is to be checked
**/
UINT is_in_sbf(UINT val, sbf* bl) {
	UINT chunk_per_ind = bloomParam->NUM_BITS/bloomParam->COUNTER_CHUNK;
	for (int i = 0; i < bloomParam->K; i++) {

		UINT a = hash(val, i);
		UINT loc = a / chunk_per_ind;
		UINT off = a % chunk_per_ind;

		UINT bitmap = (bloomParam->BIT_MAP<<(off * bloomParam->COUNTER_CHUNK));
		UINT chunk = bitmap & bl->flag_array[loc];
		if (!chunk) return 0;
	}

	return 1;
}

/**
	Inserts a value in an SBF. If the value is already present in the SBF, it does nothing.
	@param val 	: The value to be inserted
	@param bl 	: The SBF in which the value is to be inserted
**/
bool insert_in_sbf(UINT val, sbf* bl) {
	UINT i;
	UINT chunk_per_ind = bloomParam->NUM_BITS / bloomParam->COUNTER_CHUNK;

	if(is_in_sbf(val, bl)) {
		return false;
	}

	bl->num_elems++;
	
	for (i = 0; i < bloomParam->K; i++) {

		UINT a = hash(val, i);
		UINT loc = a / chunk_per_ind;
		UINT off = a % chunk_per_ind;

		// Sidharth: modified 1 to 1ULL
		UINT new_ = 1ULL << (off * bloomParam->COUNTER_CHUNK);
		UINT bitmap = (bloomParam->BIT_MAP<<(off * bloomParam->COUNTER_CHUNK));
		UINT chunk = bitmap & bl->flag_array[loc];
		if(chunk == bitmap) {

			// printf("overflow of the bucket for counter -- making no change\n");
			continue;
		}

		bl->flag_array[loc] = bl->flag_array[loc] + new_;
	}

	return true;
}

/**
	Counts the number of ones in a "chunk" of a given integer when represented in binary form.
	For example, if the chunk is 5, it'll count the number of ones in the first 5 bits of the number.
	The chunk size is calculated using bloomParams. 
	Used by num_ones_in_sbf() .
	@param x 		: The integer in which the number of ones is to be calculated
	@return count 	: The number of ones in the "chunk" of x	 
**/
UINT count_ones(UINT x) {
	UINT i, count = 0;
	for(i = 0; i < bloomParam->NUM_BITS/bloomParam->COUNTER_CHUNK; i++) {

		UINT new_ = bloomParam->BIT_MAP<<(i * bloomParam->COUNTER_CHUNK);
		UINT intersect = x & new_;
		count += intersect>>(i * bloomParam->COUNTER_CHUNK);
	}

	return count;
}

/**
	Counts the number of ones in the bit array of an SBF
	@param a 		: The SBF in which the number of ones is to be counted
	@return count 	: The number of ones in the SBF 
**/
UINT num_ones_in_sbf(sbf *a) {
	UINT i, n = 0;
	for (i = 0; i < bloomParam->SIZE_BLOOM / bloomParam->NUM_BITS; i++) {
		n += count_ones(a->flag_array[i]);
	}
		
	return n;
}

/**
	Approximates the number of elements that could be present in an SBF as given by:
	"https://en.wikipedia.org/wiki/Bloom_filter#Approximating_the_number_of_items_in_a_Bloom_filter"
	@param a 		: The SBF in which the number of elements is to be counted
	@return count 	: The approximated number of elements in a  
**/
UINT count_elems_in_sbf(sbf *a) {
	UINT ret;
	double estimated_count = (1.0 * bloomParam->SIZE_BLOOM) / bloomParam->K;
	estimated_count *= -1;
	estimated_count *= log(1.0 - (1.0 * num_ones_in_sbf(a)) / bloomParam->SIZE_BLOOM);
	ret = ceil(estimated_count);
	return ret;
}

/**
	Makes a copy of an SBF. Required while updating the snapshot 
	@param a : The SBF in which the copy is to be stored
	@param b : The SBF which is to be copied
**/
void copy_sbf(sbf *a, sbf *b) {
	a->num_elems = b->num_elems;

	// Copy the bits
	for(int i=0; i < bloomParam->SIZE_BLOOM / bloomParam->NUM_BITS; i++) {
		(a->flag_array)[i] = (b->flag_array)[i];
	}
}

/**
	Helper function used by sbf_intersect(). Doesn't know why it is required.
	@param a,b 	: The SBF's whose intersection is to be taken
	@param c 	: The SBF in which the intersection is to be stored
**/
void intersect_bloom_count_node(sbf *a, sbf *b, sbf *c) {
	UINT i, j;
	for(i = 0; i< (bloomParam->SIZE_BLOOM / bloomParam->NUM_BITS); i++) {

		UINT temp = 0;
		for(j = 0; j < bloomParam->NUM_BITS / bloomParam->COUNTER_CHUNK; j++) {

			UINT new_ = bloomParam->BIT_MAP<<(j * bloomParam->COUNTER_CHUNK);
			temp += MIN(a->flag_array[i] & new_, b->flag_array[i] & new_);
		}

		c->flag_array[i] = temp;
	}
}

/**
	Helper function used by sbf_union(). Doesn't know why it is required.
	@param a,b 	: The SBF's whose union is to be taken
	@param c 	: The SBF in which the union is to be stored
**/
void union_bloom_count_node(sbf *a, sbf *b, sbf *c) {
	UINT i, j;
	for(i = 0; i< (bloomParam->SIZE_BLOOM / bloomParam->NUM_BITS); i++) {

		UINT temp = 0;
		for(j = 0; j < bloomParam->NUM_BITS / bloomParam->COUNTER_CHUNK; j++) {

			UINT new_ = bloomParam->BIT_MAP<<(j*bloomParam->COUNTER_CHUNK);
			temp += MAX(a->flag_array[i] & new_, b->flag_array[i] & new_);
		}

		c->flag_array[i] = temp;
	}
}

/**
	Computes the intersection of two SBF's 
	@param a,b 	: The SBF's whose intersection is to be taken
	@param c 	: The SBF in which the intersection is stored
**/
void intersect_sbf(sbf *a, sbf *b, sbf *c) {
	if(bloomParam->COUNTER_CHUNK > 1) {

		printf("Please set the bin size to one!!\n");
		intersect_bloom_count_node(a, b, c);
	}

	for(int i = 0; i < bloomParam->SIZE_BLOOM / bloomParam->NUM_BITS; i++) {

		UINT temp = a->flag_array[i] & b->flag_array[i];
		c->flag_array[i] = temp;
	}

	c->num_elems = count_elems_in_sbf(c);
	if(bloomParam->FALSE_PSTV_PROB < c->num_elems) {
		c->num_elems = bloomParam->FALSE_PSTV_PROB;
	}
}

/**
	Computes the union of two SBF's 
	@param a,b 	: The SBF's whose union is to be taken
	@param c 	: The SBF in which the union is stored
**/
void union_sbf(sbf *a, sbf *b, sbf *c) {
	if(bloomParam->COUNTER_CHUNK > 1) {

		printf("Please set the bin size to one!!\n");
		union_bloom_count_node(a, b, c);
	}

	for(int i = 0; i < bloomParam->SIZE_BLOOM / bloomParam->NUM_BITS; i++) {

		UINT temp = a->flag_array[i] | b->flag_array[i];
		c->flag_array[i] = temp;
	}
	
	c->num_elems = count_elems_in_sbf(c);
	if(bloomParam->FALSE_PSTV_PROB < c->num_elems) {
		c->num_elems = bloomParam->FALSE_PSTV_PROB;
	}
}

#endif