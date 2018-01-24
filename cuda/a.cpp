#include <cstdio>
#include <cstring>
void calc(int *a){
	printf("%d\n",*a);
}
int main(){
	int a[10],b[10];
	for(int i=0;i<10;i++ )a[i] = i,b[i]=0;
	memcpy(b,a,40);	

	calc(a+2);


}