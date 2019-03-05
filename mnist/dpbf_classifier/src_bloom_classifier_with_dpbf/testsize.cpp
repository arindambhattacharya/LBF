#include <iostream>

using namespace std;

struct p{
	int n;
	int *a;
}
;
typedef struct p p;

int main()
{

	p x;
	x.a = new int[25];
	int *m = new int[487];
	cout << sizeof(x.n) << endl;
	cout << sizeof(m) << endl;

	return 0;
}