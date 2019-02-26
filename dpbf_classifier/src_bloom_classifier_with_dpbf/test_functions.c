#include <iostream>
#include <fstream>
#include <ctime>
#include <stdlib.h>
#include <string.h>
#include <iomanip>
#include "dpbf.h"

using namespace std;

char *MEMORY_TEST_FILENAME, *FP_RATE_TEST_FILENAME, *QUERY_TIME_TEST_FILENAME, *CONSTRUCTION_TIME_TEST_FILENAME, *UNION_TEST_FILENAME, *INTERSECTION_TEST_FILENAME, *RATIO_TEST_FILENAME, *ALL_TESTS_FILENAME, *PARAMETER_PLOTTER_FILENAME;

// Reads twitter data and returns an array storing the same
UINT* read_data_from_file(char* filename, long long int sample_size)
{
	UINT *selected = (unsigned long long int *)malloc(sample_size*sizeof(unsigned long long int));
	FILE *reader = fopen(filename, "r");
	for(long long int i = 0; i < sample_size; i++) {
		fscanf(reader, "%lld\n", &selected[i]);
	}

	fclose(reader);
	return selected;
}

// Test for query time
void query_time_test(UINT *userIDs, double fp_prob) {

	// Open the output file
	ofstream query_time_test_file;
	query_time_test_file.open(QUERY_TIME_TEST_FILENAME);

	// Declare the structure
	dpbf *a;

	int plot_points[18] = {0,10,20,60,100,200,600,1000,2000,6000,10000,20000,60000,100000,200000,600000,1000000,2000000};

	// Initialize bloom parameters
	int NUM_ITERATIONS = 1;
	UINT M = (1ULL << 34);
	UINT size_bloom = 1024;
	UINT k = 3;
	UINT counter_chunk = 1;
	UINT num_partition = 30;
	
 	init_bloomParameters(size_bloom, k, counter_chunk, num_partition, fp_prob);
 	seive_initial();

	// Initialize
	a = (dpbf*) malloc(sizeof(dpbf));
	init_dpbf(a);

	for(int idx = 0; idx < 17; idx++){
		double avg_qt = 0.0;

		// Insert elements
	 	for(int i = plot_points[idx]; i < plot_points[idx + 1]; i++){
			insert_in_dpbf(userIDs[i], a);
		}

		// Update the snapshot
		a->snapshot = update_snapshot(a->snapshot, 0, 0, a->myMap);

		for(int iteration = 0; iteration < NUM_ITERATIONS; iteration++){
			int num_queries = 1e6;
			clock_t start, end;

			start = clock();

			// Check query time
			for(int i = 0; i < num_queries; i++){
				is_in_dpbf(userIDs[i], a);
			}

			end = clock();
			
			avg_qt += ((double) (end - start)) / CLOCKS_PER_SEC;
		}

		// Output to file
		query_time_test_file << plot_points[idx + 1] << " " << avg_qt / NUM_ITERATIONS << endl;
	}

	// Free memory
	free_dpbf(a);

	// Close the output file
	query_time_test_file.close();

}

// Test for construction time
void construction_time_test(double fp_prob) {
	
	// Open the output file
	ofstream construction_time_test_file;
	construction_time_test_file.open(CONSTRUCTION_TIME_TEST_FILENAME);

	// Declare the structure
	dpbf *a;

	// Initialize bloom parameters
	int NUM_ITERATIONS = 1e3;
	UINT M = (1ULL << 34);
	UINT size_bloom = 1024;
	UINT k = 3;
	UINT counter_chunk = 1;
	UINT num_partition = 30;

 	init_bloomParameters(size_bloom, k, counter_chunk, num_partition, fp_prob);
 	seive_initial();

	double avg_ct = 0.0;
	clock_t start, end;

	for(int iteration = 0; iteration < NUM_ITERATIONS; iteration++){
		start = clock();
		
		// Initialize
		a = (dpbf*) malloc(sizeof(dpbf));
		init_dpbf(a);

		end = clock();

		avg_ct += ((double) (end - start)) / CLOCKS_PER_SEC;
		
		construction_time_test_file<<((double) (end - start)) / CLOCKS_PER_SEC<<endl;

		// Free memory
		free_dpbf(a);

	}

	construction_time_test_file<<"Average time : "<<avg_ct / NUM_ITERATIONS<<endl;

	// Close the output file
	construction_time_test_file.close();

}

// Test for total memory
void memory_test(UINT *userIDs, double fp_prob) {

	// Open the output file
	ofstream memory_test_file;
	memory_test_file.open(MEMORY_TEST_FILENAME);

	// Declare the structure
	dpbf *a;

	int plot_points[18] = {0,10,20,60,100,200,600,1000,2000,6000,10000,20000,60000,100000,200000,600000,1000000,2000000};

	// Initialize bloom parameters
	UINT M = (1ULL << 34);
	UINT size_bloom = 1024;
	UINT k = 3;
	UINT counter_chunk = 1;
	UINT num_partition = 30;
 	init_bloomParameters(size_bloom, k, counter_chunk, num_partition, fp_prob);
 	seive_initial();

	long long memory_used = 0;
		
	// Initialize
	a = (dpbf*) malloc(sizeof(dpbf));
	init_dpbf(a);

	for(int idx = 0; idx < 17; idx++){

		// Insert elements
	 	for(int i = plot_points[idx]; i < plot_points[idx + 1]; i++){
			insert_in_dpbf(userIDs[i], a);
		}

		// Update the snapshot
		a->snapshot = update_snapshot(a->snapshot, 0, 0, a->myMap);

		// Calculate memory used (in bytes)
		// Memory of snapshot
		memory_used = count_leaves(a->snapshot) * (sizeof(b_node) + (bloomParam->SIZE_BLOOM / 8)) + count_non_leaves(a->snapshot) * sizeof(b_node);
		
		// // Memory of map
		// memory_used += ;

		memory_test_file<<plot_points[idx + 1]<<" "<<memory_used<<" "<<count_leaves(a->snapshot)<<endl;

	}

	// Free memory
	free_dpbf(a);

	// Close the output file
	memory_test_file.close();

}

// Test for fp rate
void fp_rate_test(UINT *userIDs, double fp_prob, long long reduced_namespace_size, UINT* reduced_namespace) {

	// Open the output file
	ofstream fp_rate_test_file;
	fp_rate_test_file.open(FP_RATE_TEST_FILENAME);

	// Declare the structure
	dpbf *a;

	int plot_points[18] = {0,10,20,60,100,200,600,1000,2000,6000,10000,20000,60000,100000,200000,600000,1000000,2000000};
	int true_positives[18] = {0,0,0,0,0,0,3,6,14,60,110,212,620,998,1974,5949,10061,20059};

	// Initialize bloom parameters
	UINT M = (1ULL << 34);
	UINT size_bloom = 1024;
	UINT k = 3;
	UINT counter_chunk = 1;
	UINT num_partition = 30;
 	init_bloomParameters(size_bloom, k, counter_chunk, num_partition, fp_prob);
 	seive_initial();
	
	// Initialize
	a = (dpbf*) malloc(sizeof(dpbf));
	init_dpbf(a);

	for(int idx = 0; idx < 17; idx++){
		double avg_fp = 0.0;
		
		// Insert elements
	 	for(int i = plot_points[idx]; i < plot_points[idx + 1]; i++){
			insert_in_dpbf(userIDs[i], a);
		}

		// Update the snapshot
		a->snapshot = update_snapshot(a->snapshot, 0, 0, a->myMap);

		// Calculate fp_rate
		int positives = 0;
		for(int i=0; i < reduced_namespace_size; i++) {
			if(is_in_dpbf(reduced_namespace[i], a)) {
				positives++;
			}
		}

		avg_fp += (double)(positives - true_positives[idx + 1]) / (double)(reduced_namespace_size);

		fp_rate_test_file<<plot_points[idx + 1]<<" "<<avg_fp<<endl;
	}

	// Free memory
	free_dpbf(a);

	// Close the output file
	fp_rate_test_file.close();

}

// Test for set union
void set_union_test(UINT *userIDs, double fp_prob, long long reduced_namespace_size, UINT* reduced_namespace) {
	
	// Open the output file
	ofstream union_test_file;
	union_test_file.open(UNION_TEST_FILENAME);

	// Declare the structures
	dpbf *a, *b, *c;

	int plot_points[8] = {0,1000,2000,6000,10000,20000,60000,100000};
	int true_positives[9] = {0,48,60,72,118,152,263,668,1049};
	int num_elements_in_intersection = 5000, previous_index = 5000, diff;

	// Initialize
	a = (dpbf*) malloc(sizeof(dpbf));
	b = (dpbf*) malloc(sizeof(dpbf));
	init_dpbf(a);
	init_dpbf(b);

	// Initialize bloom parameters
	UINT M = (1ULL << 34);
	UINT size_bloom = 1024;
	UINT k = 3;
	UINT counter_chunk = 1;
	UINT num_partition = 30;
	
 	init_bloomParameters(size_bloom, k, counter_chunk, num_partition, fp_prob);
 	seive_initial();

 	// Insert common elements into a and b
 	for(int i=0; i < num_elements_in_intersection; i++) {
 		insert_in_dpbf(userIDs[i], a);
 		insert_in_dpbf(userIDs[i], b);
 	}

	for(int idx = 1; idx <= 7; idx++){

		int diff = plot_points[idx] - plot_points[idx - 1];
		int set_size = diff / 2;
		double avg_union_fp = 0.0;

		// Insert elements in a
	 	for(int i = previous_index; i < previous_index + set_size; i++){
			insert_in_dpbf(userIDs[i], a);
		}

		previous_index += set_size;

		// Insert elements in b
		for(int i = previous_index; i < previous_index + set_size; i++) {
			insert_in_dpbf(userIDs[i], b);
		}

		previous_index += set_size;

		// Find union
		c = union_dpbf(a, b);

		// Calculate fp_rate
		UINT positives = 0;
		for(int i=0; i < reduced_namespace_size; i++) {
			if(is_in_dpbf(reduced_namespace[i], c)) {
				positives++;
			}
		}

		avg_union_fp += (double)(positives - true_positives[idx + 1]) / (double)(reduced_namespace_size);

		union_test_file<<plot_points[idx]<<" "<<avg_union_fp<<endl;

		// Free memory
		free_dpbf(c);
	}

	free_dpbf(a);
	free_dpbf(b);

	// Close the output file
	union_test_file.close();
}

// Test for set union
void set_intersection_test(UINT *userIDs, double fp_prob, long long reduced_namespace_size, UINT* reduced_namespace) {
	
	// Open the output file
	ofstream intersection_test_file;
	intersection_test_file.open(INTERSECTION_TEST_FILENAME);

	// Declare the structures
	dpbf *a, *b, *c;

	int plot_points[8] = {0,1000,2000,6000,10000,20000,60000,100000};
	int true_positives[9] = {0,48,60,72,118,152,263,668,1049};
	int num_elements_in_intersection = 5000, previous_index = 5000, diff;

	// Initialize
	a = (dpbf*) malloc(sizeof(dpbf));
	b = (dpbf*) malloc(sizeof(dpbf));
	init_dpbf(a);
	init_dpbf(b);

	// Initialize bloom parameters
	UINT M = (1ULL << 34);
	UINT size_bloom = 1024;
	UINT k = 3;
	UINT counter_chunk = 1;
	UINT num_partition = 21;
	
 	init_bloomParameters(size_bloom, k, counter_chunk, num_partition, fp_prob);
 	seive_initial();

 	// Insert common elements into a and b
 	for(int i=0; i < num_elements_in_intersection; i++) {
 		insert_in_dpbf(userIDs[i], a);
 		insert_in_dpbf(userIDs[i], b);
 	}

	for(int idx = 1; idx <= 7; idx++){

		int diff = plot_points[idx] - plot_points[idx - 1];
		int set_size = diff / 2;
		double avg_intersection_fp = 0.0;

		// Insert elements in a
	 	for(int i = previous_index; i < previous_index + set_size; i++){
			insert_in_dpbf(userIDs[i], a);
		}

		previous_index += set_size;

		// Insert elements in b
		for(int i = previous_index; i < previous_index + set_size; i++) {
			insert_in_dpbf(userIDs[i], b);
		}

		previous_index += set_size;

		// Find intersection
		c = intersection_dpbf(a, b);

		// Calculate fp_rate
		UINT positives = 0;
		for(int i=0; i < reduced_namespace_size; i++) {
			if(is_in_dpbf(reduced_namespace[i], c)) {
				positives++;
			}
		}

		avg_intersection_fp += (double)(positives - true_positives[1]) / (double)(reduced_namespace_size);

		intersection_test_file<<plot_points[idx]<<" "<<avg_intersection_fp<<endl;

		// Free memory
		free_dpbf(c);
	}

	free_dpbf(a);
	free_dpbf(b);

	// Close the output file
	intersection_test_file.close();
}

void ratio_test(UINT* userIDs, double fp_prob) {

	// Open the output file
	ofstream ratio_test_file;
	ratio_test_file.open(RATIO_TEST_FILENAME);

	// Declare the structure
	dpbf *a;

	// Initialize bloom parameters
	UINT NUM_ITERATIONS = 1000;
	UINT M = (1ULL << 34);
	UINT size_bloom = 1024;
	UINT k = 3;
	UINT counter_chunk = 1;
	UINT num_partition = 30;
	clock_t start, end;
	
 	init_bloomParameters(size_bloom, k, counter_chunk, num_partition, fp_prob);
 	seive_initial();

 	for(double ratio = 0.001; ratio <= 1000; ratio = ratio * 10) {

 		// Initialize
		a = (dpbf*) malloc(sizeof(dpbf));
		init_dpbf(a);

		srand(time(NULL));

		int query_num = 10000, insert_num = (int)((double)query_num * ratio);

		double avg_time = 0.0;
	 	for(UINT itr = 0; itr < NUM_ITERATIONS; itr++) {

	 		for(int k = 0; k < insert_num; k++) {
		 		insert_in_dpbf(userIDs[k + (itr * insert_num)], a); 			
	 		}
	
	 		start = clock();

	 		for(int k = 0; k < query_num; k++) {
	 			is_in_dpbf(rand() % 10000000000, a);
	 		}

	 		end = clock();
	 		avg_time += (double)(end - start) / (CLOCKS_PER_SEC);

	 	}

	 	ratio_test_file<<ratio<<" "<<avg_time / (query_num * NUM_ITERATIONS)<<endl;

		// Free memory
		free_dpbf(a);
 	}

	// Close the output file
	ratio_test_file.close();
}

// void all_tests(UINT *userIDs, double fp_prob, long long reduced_namespace_size, UINT* reduced_namespace) {
void all_tests(UINT *userIDs, double fp_prob) {

	// Open the output file
	ofstream all_tests_file;
	all_tests_file.open(ALL_TESTS_FILENAME);

	// Declare the structure
	dpbf *a;

	int plot_points[18] = {0,10,20,60,100,200,600,1000,2000,6000,10000,20000,60000,100000,200000,600000,1000000,2000000};
	int true_positives[18] = {0,0,0,0,0,0,3,6,14,60,110,212,620,998,1974,5949,10061,20059};

	// Initialize bloom parameters
	int NUM_ITERATIONS = 1;
	UINT M = (1ULL << 34);
	UINT size_bloom = 1024;
	UINT k = 3;
	UINT counter_chunk = 1;
	UINT num_partition = 30;
	
 	init_bloomParameters(size_bloom, k, counter_chunk, num_partition, fp_prob);
 	seive_initial();

	double construction_time = 0.0, query_time = 0.0, fp_rate = 0.0, insertion_time = 0.0, updation_time = 0.0;
	UINT memory_snapshot = 0, memory_pbf = 0, positives, memory_used;

	clock_t start, end;
	double insert_time;

	for(int idx = 16; idx < 17; idx++){

		printf("construction time\n");
		// Construction time test 
		{
			// Initialize
			a = (dpbf*) malloc(sizeof(dpbf));
			init_dpbf(a);

			start = clock();

			// Insert elements
		 	for(int i = 0; i < plot_points[idx]; i++){
				insert_in_dpbf(userIDs[i], a);
			}

			end = clock();

			insertion_time = (double)(end - start) / CLOCKS_PER_SEC;

			start = clock();

			// Update the snapshot
			a->snapshot = update_snapshot(a->snapshot, 0, 0, a->myMap);

			end = clock();

			updation_time = (double)(end - start) / CLOCKS_PER_SEC;
			construction_time = insertion_time + updation_time;
		}

		/*
		printf("memory\n");
		// Memory test 
		{
			// Memory of snapshot
			memory_snapshot = count_leaves(a->snapshot) * (sizeof(b_node) + (bloomParam->SIZE_BLOOM / 8)) + count_non_leaves(a->snapshot) * sizeof(b_node);
			
			// // Memory of linked list
			// memory_pbf = count_nodes_in_linked_list(&(a->pbf_list)) * (sizeof(bf_linked_list_node));

			memory_used = memory_snapshot + memory_pbf;
		}
		
		printf("query time\n");
		// Query time test
		{
			for(int iteration = 0; iteration < NUM_ITERATIONS; iteration++){
				int num_queries = 1e6;

				start = clock();

				// Check query time
				for(int i = 0; i < num_queries; i++){
					is_in_dpbf(userIDs[i], a);
				}

				end = clock();
				
				query_time += ((double) (end - start)) / CLOCKS_PER_SEC;
			}

			query_time /= NUM_ITERATIONS;
		}

		printf("fp rate\n");
		// FP rate test 
		{
			positives = 0;
			for(UINT i=0; i < reduced_namespace_size; i++) {
				if(is_in_dpbf(reduced_namespace[i], a)) {
					positives++;
				}
			}

			fp_rate = (double)(positives - true_positives[idx]) / (double)(reduced_namespace_size);
		}
		*/	

		// Output to file
		// all_tests_file << plot_points[idx] << " " << construction_time << " " << memory_snapshot << " " << memory_pbf << " " << memory_used << " " << query_time << " " << fp_rate << " " << positives << endl;
		all_tests_file << plot_points[idx] << " " << insertion_time << " " << updation_time << " " << construction_time << endl;

		// Free memory
		free_dpbf(a);
	}

	// Close the output file
	all_tests_file.close();
}

void parameter_plotter(UINT* userIDs, double fp_prob) {

	// Open the output file
	ofstream parameter_plotter_file;
	parameter_plotter_file.open(PARAMETER_PLOTTER_FILENAME);

	// Declare the structure
	dpbf *a;

	// Initialize bloom parameters
	UINT M = (1ULL << 34);
	UINT k = 3;
	UINT counter_chunk = 1;
	double inv_k = (double)1 / k;
	double p_1_k = pow(fp_prob, inv_k);
	double inv_log = (double)1 / (log(1.0 - p_1_k));
	UINT SET_SIZE = 100000, start, end;

	for(int d = 30; d < 31; d++) {
		UINT size_bloom =  k * (-1 * inv_log) * pow(2, 34 - d);
		init_bloomParameters(size_bloom, k, counter_chunk, d, fp_prob);
	 	seive_initial();

		cout<<"size bloom : "<<size_bloom<<endl;

		UINT memory_snapshot = 0, memory_pbf = 0;
		double construction_time;

		start = clock();

		// Initialize
		a = (dpbf*) malloc(sizeof(dpbf));
		init_dpbf(a);

		printf("inserting\n");

		// Insert elements
	 	for(int i = 0; i < SET_SIZE; i++){
			insert_in_dpbf(userIDs[i], a);
		}

		printf("updating\n");
		// Update the snapshot
		a->snapshot = update_snapshot(a->snapshot, 0, 0, a->myMap);

		end = clock();

		construction_time = (double)(end - start) / CLOCKS_PER_SEC;

		// Calculate memory used (in bytes)
		// Memory of snapshot
		memory_snapshot = count_leaves(a->snapshot) * (sizeof(b_node) + (bloomParam->SIZE_BLOOM / 8)) + count_non_leaves(a->snapshot) * sizeof(b_node);
		
		// // Memory of linked list
		// memory_pbf = count_nodes_in_linked_list(&(a->pbf_list)) * sizeof(bf_linked_list_node);		

		parameter_plotter_file << d << " " << size_bloom << " " << memory_snapshot << " " << memory_pbf << " " << construction_time <<endl;

	}

	// Free memory
	free_dpbf(a);

	// Close the output file
	parameter_plotter_file.close();	

}

// New Structure
int main() {

	// Declare and initialize filenames
	MEMORY_TEST_FILENAME = (char*)malloc(sizeof(char) * 50);
	FP_RATE_TEST_FILENAME = (char*)malloc(sizeof(char) * 50);
	QUERY_TIME_TEST_FILENAME = (char*)malloc(sizeof(char) * 50);
	CONSTRUCTION_TIME_TEST_FILENAME = (char*)malloc(sizeof(char) * 50);
	UNION_TEST_FILENAME = (char*)malloc(sizeof(char) * 50);
	INTERSECTION_TEST_FILENAME = (char*)malloc(sizeof(char) * 50);
	RATIO_TEST_FILENAME = (char*)malloc(sizeof(char) * 50);
	ALL_TESTS_FILENAME = (char*)malloc(sizeof(char) * 50);
	PARAMETER_PLOTTER_FILENAME = (char*)malloc(sizeof(char) * 50);


	strcpy(MEMORY_TEST_FILENAME, "../output/memory_test.txt");
	strcpy(FP_RATE_TEST_FILENAME, "../output/fp_rate_test.txt");
	strcpy(QUERY_TIME_TEST_FILENAME, "../output/query_time_test.txt");
	strcpy(CONSTRUCTION_TIME_TEST_FILENAME, "../output/construction_time_test.txt");
	strcpy(UNION_TEST_FILENAME, "../output/union_test.txt");
	strcpy(INTERSECTION_TEST_FILENAME, "../output/intersection_test.txt");
	strcpy(RATIO_TEST_FILENAME, "../output/ratio_test.txt");
	strcpy(ALL_TESTS_FILENAME, "../output/all_tests.txt");
	strcpy(PARAMETER_PLOTTER_FILENAME, "../output/parameter_plotter.txt");

	// Read twitter data
	UINT* userIDs = read_data_from_file("../../../../../data/twitter_data/unique_userIDs.txt", 3008497);
	// UINT* reduced_namespace = read_data_from_file("../../../../../data/twitter_data/random_sample.txt", 1e8);

	// Initialize fp_threshold
	double fp_threshold = 1e-4;

	// // Run tests
	// memory_test(userIDs, fp_threshold);
	// construction_time_test(fp_threshold);
	// query_time_test(userIDs, fp_threshold);
	// fp_rate_test(userIDs, fp_threshold, 1e8, reduced_namespace);
	// ratio_test(userIDs, fp_threshold);
	// set_union_test(userIDs, fp_threshold, 1e8, reduced_namespace);
	// set_intersection_test(userIDs, fp_threshold, 1e8, reduced_namespace);
	// all_tests(userIDs, fp_threshold, 1e8, reduced_namespace);
	all_tests(userIDs, fp_threshold);
	// parameter_plotter(userIDs, fp_threshold);

	return 0;
}
