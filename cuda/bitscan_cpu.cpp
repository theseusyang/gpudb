#include<iostream>
#include<cstdio>
#include<cstring>
using namespace std;

int main(int argc, char ** argv){
	freopen("scan.in","r",stdin);
	freopen("scan.out","w",stdout);

	int n = 32;
	scanf("%d",&n);

	int bytes = n * sizeof(int);

	int *a;
	a = (int*)malloc(bytes );   


	for (int i = 0; i < n; i++)
		scanf("%d",a+i);
	int constC=0;
	scanf("%d", &constC);
	for(int i=0;i<n;i++)
		if(a[i]<constC) printf("%d\n",1);
	else printf("%d\n",0);
}