#include<vector>
#include<iostream>
#include<algorithm>
using namespace std;
const int know_stop_size = 100000 + 10;

vector<int > know_stop_num[know_stop_size], know_stop_len[know_stop_size];
int nlz(unsigned x){
   int n;

   if (x == 0) return(32);
   n = 1;
   if ((x >> 16) == 0) {n = n +16; x = x <<16;}
   if ((x >> 24) == 0) {n = n + 8; x = x << 8;}
   if ((x >> 28) == 0) {n = n + 4; x = x << 4;}
   if ((x >> 30) == 0) {n = n + 2; x = x << 2;}
   n = n - (x >> 31);
   return n;
}
int main(){
	int N = 100,early_size = 10;
	srand(time(0));
	for (int i = 0; i < N; i++) {
		know_stop_num[i/early_size].push_back (rand()%N);
	} 	
	

	for	(int i = 0;i < N/ early_size; i++){
		know_stop_num[i].push_back(0);
		know_stop_num[i].push_back((1U<<31)-1);
	  	sort(know_stop_num[i].begin(),know_stop_num[i].end());
		know_stop_num[i].erase(unique(know_stop_num[i].begin(), know_stop_num[i].end()), know_stop_num[i].end());
	  	know_stop_len[i].push_back(0);
	  	for(int j=1;j < know_stop_num[i].size();j++)
	  		know_stop_len[i].push_back(nlz(know_stop_num[i][j] ^ know_stop_num[i][j-1]));

	}
	for	(int i = 0;i < N/ early_size; i++){
		for(int j=0;j < know_stop_num[i].size();j++)
	  		printf("%d %d\n",know_stop_num[i][j],know_stop_len[i][j]);
	 }
	
}

