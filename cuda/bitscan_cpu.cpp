#include <iostream>
#include <cstdio>
#include <cstring>
#include <cstdlib>
using namespace std;
#define utype long long
int main(int argc, char ** argv){
	freopen("scan.in","r",stdin);
	freopen("scan.out","w",stdout);

	int n = 32;
	scanf("%d",&n);

	utype bytes = n * sizeof(utype);

	utype *a;
	a = (utype*)malloc(bytes );   


	for (int i = 0; i < n; i++)
		if(sizeof(utype)==4)scanf("%d",a+i);
		else scanf("%lld",a+i);
	utype constC=0;
	int test_num = 0;
	scanf("%d",&test_num);
	for(int i = 1;i<=test_num;i++)
	if(sizeof(utype)==4)scanf("%d", &constC);
	else scanf("%lld",&constC);
	for(int i=0;i<n;i++)
		if(a[i]<constC) printf("%d\n",1);
	else printf("%d\n",0);
}