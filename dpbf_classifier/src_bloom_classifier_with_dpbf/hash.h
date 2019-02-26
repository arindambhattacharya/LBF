#ifndef _HASH_H_
#define _HASH_H_

#include "bloom_parameters.h"

/***
	Class for storing the hash functions used by SBF. Could be merged with sbf.h but created a 
	separate file so that the function can be changed easily. Must have the following functions - 

	hash(UINT a, UINT i)
***/

int aHash[10]={3730, 9830, 3830, 2110, 4740, 9440, 5540, 2710, 690, 7470};
int bHash[10]={269, 229, 929, 949, 624, 390, 64, 67, 945, 253};
int m[10];

/**
	Initializes the hash functions using Seive of Erathostenes? Call this function before starting any tests. 
	Not calling it in initializeBloomParameters() to maintain modularity. 
	Can probably be removed if other hash functions are used.
**/
void seive_initial() {

	int num_bits = bloomParam->SIZE_BLOOM/bloomParam->COUNTER_CHUNK;
	int *x = (int*)malloc(sizeof(int)*num_bits),i,j;

	for (i=0;i<num_bits;i++) {
		x[i] = i+1;
	}
	
	for (i=1;i<num_bits;i++) {
		j = x[i];
		
		if (j==-1) {
			continue;
		}
		
		int q = 1;
		while ((i + q*j) < num_bits) {
			x[i+q*j] = -1;
			q++;
		}
	}

	i = num_bits;

	for (j=0;j<10;j++) {
		for (i=i-1;i>0;i--) {
			if (x[i]!=-1) {
				m[j] = x[i];
				break;
			}
		}
	}
}

/**
	The main hash function used by the SBF. 
	@param a 		: The value which is to be hashed
	@param i 		: Number indicating which hash functionis to be used (from 1 to k)
	@return retVal	: The hashed index in the array
**/
UINT hash(UINT a, UINT i) {
	unsigned long z = aHash[i];
	z = z*a + bHash[i];
	UINT ret_val = (z % m[i]);
	return ret_val;
}

#endif