#include<iostream>
#include<cstdio>
#include<cstring>
using namespace std;
long long ran(){
	long long x= rand();
	long long y= rand();
	return (x<<31) + y;
}
int main(int argc, char ** argv){

	freopen("scan.in","w",stdout);

	int n = 32;
	sscanf(argv[1],"%d",&n);
	n  = 1<<n;
	int bytes = n * sizeof(int);

		srand(0);

	printf("%d\n",n);
	for(int i = 0; i < n; i++)
		printf("%lld\n",ran());
	int test_num = 500;
	printf("%d\n", test_num);
	for(int i = 0;i < test_num; i++)
		printf("%lld\n", ran());

}