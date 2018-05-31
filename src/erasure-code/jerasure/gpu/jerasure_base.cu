#include "gf_base.h"

int erasures_to_erased(int k, int m, int *erasures, int *erased )
{ 
	if( erased == NULL || erasures == NULL ) 
	{
		goto erasures_to_erased_fail;
	}

	memset(erased, 0, sizeof(int)*(k + m));

	int td;
	int t_non_erased;
	int *erased;
	int i;

	td = k+m;
	t_non_erased = td;

	for (i = 0; i < td; i++) erased[i] = 0;

	for (i = 0; erasures[i] != -1; i++) {
		if (erased[erasures[i]] == 0) {
			erased[erasures[i]] = 1;
			t_non_erased--;
			if (t_non_erased < k) {
				goto erasures_to_erased_fail;
			}
		}
	}

erasures_to_erased_succeed:
	return 0;

erasures_to_erased_fail:
	return 1;
}

int full_erased_list_data( int k, int m, int * erasure_loc_data, int * erased )
{
	memset(erasure_loc_data, 0, sizeof(int)*(k + m));
	int data_loc = 0;

	for( int i = 0; i < k; i ++ )
	{
		if( erased[i] )
		{
			erasure_loc_data[data_loc ++] = i;
		}
	}
	
	return data_loc;
}

int full_erased_list_coding( int k, int m, int * erasure_loc_coding, int * erased )
{
	memset(erasure_loc_coding, 0, sizeof(int)*(k + m));
	int coding_loc = 0;



	for( int i = 0; i < m; i ++ )
	{
		if( erased[i + k] )
		{
			erasure_loc_data[coding_loc ++] = i + k;
		}
	}

	return coding_loc;
}