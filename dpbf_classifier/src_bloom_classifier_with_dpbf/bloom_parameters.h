#ifndef _BLOOM_PARAMETERS_H_
#define _BLOOM_PARAMETERS_H_

#include <math.h>

/**
	Stores the bloom parameters associated with the DPBF. The variable bloomParams is defined in dpbf.h
**/

typedef struct bloom_parameters bloom_parameters;
typedef unsigned long long int UINT;

struct bloom_parameters
{
	UINT NUM_BITS;
	UINT SIZE_BLOOM;
	UINT K;
	UINT COUNTER_CHUNK;
	UINT BIT_MAP;
	UINT NUM_PARTITION;				//number of partition functions
	float FALSE_PSTV_PROB;
	float FILL_THRESHOLD;	  		// This is the max number of elements that we can have so as to maintain the required false +ve rate
	/*
	* testing suite variables
	*/
	UINT TEST_RANGE;
};

extern bloom_parameters *bloomParam;

/**
	Initializes the bloom parameters
**/
void init_bloomParameters(UINT SIZE_BLOOM ,UINT K ,UINT COUNTER_CHUNK,UINT NUM_PARTITION,float FALSE_PSTV_PROB)
{
	bloomParam = (bloom_parameters*)malloc(sizeof(bloom_parameters));
	bloomParam->NUM_BITS     	= 8*sizeof(UINT);
	bloomParam->SIZE_BLOOM   	= SIZE_BLOOM;
	bloomParam->K 			  	= K;
	bloomParam->COUNTER_CHUNK  	= COUNTER_CHUNK;		//should be exponents of 2
	bloomParam->BIT_MAP 		= pow(2,COUNTER_CHUNK) -1 ;	//2^COUNTER_CHUNK - 1;

	bloomParam->FALSE_PSTV_PROB = FALSE_PSTV_PROB;
	bloomParam->FILL_THRESHOLD  = -1*((int)SIZE_BLOOM/((float)K*COUNTER_CHUNK))*log( 1 - pow(FALSE_PSTV_PROB,1.0/K));
	bloomParam->NUM_PARTITION	= NUM_PARTITION;
	bloomParam->TEST_RANGE 		= 1000;

	// Initialise the hash function
	// seive_initial();
}

#endif