/* file : dpbf.i */
  
/* name of module to use*/
%module dpbf
%{ 
    /* Every thing in this file is being copied in  
     wrapper file. We include the C header file necessary 
     to compile the interface */
    #include "dpbf.h" 
  	/*struct dpbf *mydpbf;
     variable declaration
    struct bloom_filter_linked_list full_tree;
	struct b_node *snapshot;*/
%} 
  
/* explicitly list functions and variables to be interfaced
double myvar; 
long long int fact(long long int n1); 
int my_mod(int m, int n); 
 */
%include "dpbf.h"
/* or if we want to interface all functions then we can simply 
   include header file like this -  
   %include "dpbf.h" 
*/