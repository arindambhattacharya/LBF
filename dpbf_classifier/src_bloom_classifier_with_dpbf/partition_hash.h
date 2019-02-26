#ifndef _PARTITION_HASH_H_
#define _PARTITION_HASH_H_

/**
	This file defines the partition function used by the DPBF. It only has one function named partition_hash()
**/

typedef unsigned long long int UINT;

/**
	The partition function  used by the DPBF. Simply returns the i'th bit of the number for i'th level
	@param val 		: The value which is being inserted in DPBF for which we have to decide which subtree to insert it into
				 	  using partition_hash()
	@param level 	: The current level in the DPBF tree for which the direction is to be decided (left or right)
	@return dir 	: Return either 0 (left) or 1 (right)
**/
inline int partition_hash(UINT val, UINT level) {
	
	// Switch case based implementation is marginally faster than: return (val & (1ULL<<level))>>level;
	
	switch(level) {
		case 0:
			return val & 1ULL;
		case 1:
			return (val & (1ULL<<1))>>1;
		case 2:
			return (val & (1ULL<<2))>>2;
		case 3:
			return (val & (1ULL<<3))>>3;
		case 4:
			return (val & (1ULL<<4))>>4;
		case 5:
			return (val & (1ULL<<5))>>5;
		case 6:
			return (val & (1ULL<<6))>>6;
		case 7:
			return (val & (1ULL<<7))>>7;
		case 8:
			return (val & (1ULL<<8))>>8;
		case 9:
			return (val & (1ULL<<9))>>9;
		case 10:
			return (val & (1ULL<<10))>>10;
		case 11:
			return (val & (1ULL<<11))>>11;
		case 12:
			return (val & (1ULL<<12))>>12;
		case 13:
			return (val & (1ULL<<13))>>13;
		case 14:
			return (val & (1ULL<<14))>>14;
		case 15:
			return (val & (1ULL<<15))>>15;
		case 16:
			return (val & (1ULL<<16))>>16;
		case 17:
			return (val & (1ULL<<17))>>17;
		case 18:
			return (val & (1ULL<<18))>>18;
		case 19:
			return (val & (1ULL<<19))>>19;
		case 20:
			return (val & (1ULL<<20))>>20;
		case 21:
			return (val & (1ULL<<21))>>21;
		case 22:
			return (val & (1ULL<<22))>>22;
		case 23:
			return (val & (1ULL<<23))>>23;
		case 24:
			return (val & (1ULL<<24))>>24;
		case 25:
			return (val & (1ULL<<25))>>25;
		case 26:
			return (val & (1ULL<<26))>>26;
		case 27:
			return (val & (1ULL<<27))>>27;
		case 28:
			return (val & (1ULL<<28))>>28;
		case 29:
			return (val & (1ULL<<29))>>29;
		case 30:
			return (val & (1ULL<<30))>>30;
		case 31:
			return (val & (1ULL<<31))>>31;
		case 32:
			return (val & (1ULL<<32))>>32;
		case 33:
			return (val & (1ULL<<33))>>33;
		case 34:
			return (val & (1ULL<<34))>>34;
	}

	printf("Value of level is larger than what we can handle\n");
	exit(0);
	return -1;
}

#endif