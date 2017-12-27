/*
   Copyright (c) 2012-2013 The Ohio State University.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include "scanImpl.cu"
#include "../include/common.h"
#include "../include/gpuCudaLib.h"
#ifdef HAS_GMM
	#include "gmm.h"
#else
	#define GMM_BUFFER_COW 0
#endif

#define CHECK_POINTER(p)   do {                     \
    if(p == NULL){                                  \
        perror("Failed to allocate host memory");   \
        exit(-1);                                   \
    }} while(0)

/*
 * stringCmp: Compare two strings on GPU using one single GPU thread.
 * @buf1: the first input buffer
 * @buf2: the second input buffer
 * @size: the length of data to be compared
 *
 * Return 
 *  1 if buf1 is larger
 *  0 if they are equal
 *  -1 if buf2 is larger
 */

extern char *col_buf;

__device__ static inline int stringCmp(char* buf1, char *buf2, int size){
    int i;
    int res = 0;
    for(i=0;i<size;i++){
        if(buf1[i] > buf2[i]){
            res = 1;
            break;
        }else if (buf1[i] < buf2[i]){
            res = -1;
            break;
        }
        if(buf1[i] == 0 && buf2[i] == 0)
            break;
    }
    return res;
}

/*
 * testCon: evaluate one selection predicate using one GPU thread
 * @buf1: input data to be tested
 * @buf2: the test criterion, usually a number of a string.
 * @size: the size of the input data buf1
 * @type: the type of the input data buf1 
 * @rel: >,<, >=, <= or ==.
 *
 * Return:
 *  0 if the input data meets the criteria
 *  1 otherwise
 */

__device__ static inline int testCon(char *buf1, char* buf2, int size, int type, int rel){
    int res = 1;
    if (type == INT){
        if(rel == EQ){
            res = ( *((int*)buf1) == *(((int*)buf2)) );
        }else if (rel == GTH){
            res = ( *((int*)buf1) > *(((int*)buf2)) );
        }else if (rel == LTH){
            res = ( *((int*)buf1) < *(((int*)buf2)) );
        }else if (rel == GEQ){
            res = ( *((int*)buf1) >= *(((int*)buf2)) );
        }else if (rel == LEQ){
            res = ( *((int*)buf1) <= *(((int*)buf2)) );
        }

    }else if (type == FLOAT){
        if(rel == EQ){
            res = ( *((float*)buf1) == *(((float*)buf2)) );
        }else if (rel == GTH){
            res = ( *((float*)buf1) > *(((float*)buf2)) );
        }else if (rel == LTH){
            res = ( *((float*)buf1) < *(((float*)buf2)) );
        }else if (rel == GEQ){
            res = ( *((float*)buf1) >= *(((float*)buf2)) );
        }else if (rel == LEQ){
            res = ( *((float*)buf1) <= *(((float*)buf2)) );
        }

    }else{
        int tmp = stringCmp(buf1,buf2,size);
        if(rel == EQ){
            res = (tmp == 0);
        }else if (rel == GTH){
            res = (tmp > 0);
        }else if (rel == LTH){
            res = (tmp < 0);
        }else if (rel == GEQ){
            res = (tmp >= 0);
        }else if (rel == LEQ){
            res = (tmp <= 0);
        }
    }
    return res;
}


/*
 * transform_dict_filter_and: merge the filter for dictionary-compressed predicate into the final filter.
 * @dictFilter: the filter for the dictionary compressed data
 * @dictFact: the compressed fact table column
 * @tupleNum: the number of tuples in the column
 * @filter: the filter for the uncompressed data
 */

__global__ static void transform_dict_filter_and(int * dictFilter, char *dictFact, long tupleNum, int dNum,  int * filter, int byteNum){

    int stride = blockDim.x * gridDim.x;
    int offset = blockIdx.x*blockDim.x + threadIdx.x;

    int * fact = (int*)(dictFact + sizeof(struct dictHeader));

    int numInt = (tupleNum * byteNum +sizeof(int) - 1) / sizeof(int) ; 

    for(long i=offset; i<numInt; i += stride){
        int tmp = fact[i];
        unsigned long bit = 1;

        for(int j=0; j< sizeof(int)/byteNum; j++){
            int t = (bit << ((j+1)*byteNum*8)) -1 - ((1<<(j*byteNum*8))-1);
            int fkey = (tmp & t)>> (j*byteNum*8) ;
            filter[i* sizeof(int)/byteNum + j] &= dictFilter[fkey];
        }
    }
}

__global__ static void transform_dict_filter_init(int * dictFilter, char *dictFact, long tupleNum, int dNum,  int * filter,int byteNum){

    int stride = blockDim.x * gridDim.x;
    int offset = blockIdx.x*blockDim.x + threadIdx.x;

    int * fact = (int*)(dictFact + sizeof(struct dictHeader));
    int numInt = (tupleNum * byteNum +sizeof(int) - 1) / sizeof(int) ;

    for(long i=offset; i<numInt; i += stride){
        int tmp = fact[i];
        unsigned long bit = 1;

        for(int j=0; j< sizeof(int)/byteNum; j++){
            int t = (bit << ((j+1)*byteNum*8)) -1 - ((1<<(j*byteNum*8))-1);
            int fkey = (tmp & t)>> (j*byteNum*8) ;
            filter[i* sizeof(int)/byteNum + j] = dictFilter[fkey];
        }
    }
}

__global__ static void transform_dict_filter_or(int * dictFilter, char *fact, long tupleNum, int dNum,  int * filter,int byteNum){

    int stride = blockDim.x * gridDim.x;
    int offset = blockIdx.x*blockDim.x + threadIdx.x;

    int numInt = (tupleNum * byteNum +sizeof(int) - 1) / sizeof(int) ;

    for(long i=offset; i<numInt; i += stride){
        int tmp = ((int *)fact)[i];
        unsigned long bit = 1;

        for(int j=0; j< sizeof(int)/byteNum; j++){
            int t = (bit << ((j+1)*byteNum*8)) -1 - ((1<<(j*byteNum*8))-1);
            int fkey = (tmp & t)>> (j*byteNum*8) ;
            filter[i* sizeof(int)/byteNum + j] |= dictFilter[fkey];
        }
    }
}

/*
 * genScanFilter_dict_init: generate the filter for dictionary-compressed predicate
 */

__global__ static void genScanFilter_dict_init(struct dictHeader *dheader, int colSize, int colType, int dNum, struct whereExp *where, int *dfilter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(int i=tid;i<dNum;i+=stride){
        int fkey = dheader->hash[i];
        con = testCon((char *)&fkey,where->content,colSize,colType,where->relation);
        dfilter[i] = con;
    }
}

__global__ static void genScanFilter_dict_or(struct dictHeader *dheader, int colSize, int colType, int dNum, struct whereExp *where, int *dfilter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(int i=tid;i<dNum;i+=stride){
        int fkey = dheader->hash[i];
        con = testCon((char *)&fkey,where->content,colSize,colType,where->relation);
        dfilter[i] |= con;
    }
}

__global__ static void genScanFilter_dict_and(struct dictHeader *dheader, int colSize, int colType, int dNum, struct whereExp *where, int *dfilter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(int i=tid;i<dNum;i+=stride){
        int fkey = dheader->hash[i];
        con = testCon((char *)&fkey,where->content,colSize,colType,where->relation);
        dfilter[i] &= con;
    }
}

__global__ static void genScanFilter_rle(char *col, int colSize, int colType, long tupleNum, struct whereExp *where, int andOr, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    struct rleHeader *rheader = (struct rleHeader *) col;
    int dNum = rheader->dictNum;

    for(int i = tid; i<dNum; i += stride){
        int fkey = ((int *)(col+sizeof(struct rleHeader)))[i];
        int fcount = ((int *)(col+sizeof(struct rleHeader)))[i + dNum];
        int fpos = ((int *)(col+sizeof(struct rleHeader)))[i + 2*dNum];

        con = testCon((char *)&fkey,where->content,colSize,colType,where->relation);

        for(int k=0;k<fcount;k++){
            if(andOr == AND)
                filter[fpos+k] &= con;
            else
                filter[fpos+k] |= con;
        }

    }
}

__global__ static void genScanFilter_and_eq(char *col, int colSize, long tupleNum, struct whereExp * where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = (stringCmp(col+colSize*i, where->content, colSize) == 0);
        filter[i] &= con;
    }
}

__global__ static void genScanFilter_and_gth(char *col, int colSize, long tupleNum, struct whereExp * where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = (stringCmp(col+colSize*i, where->content, colSize) > 0);
        filter[i] &= con;
    }
}

__global__ static void genScanFilter_and_lth(char *col, int colSize, long tupleNum, struct whereExp * where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = (stringCmp(col+colSize*i, where->content, colSize) < 0);
        filter[i] &= con;
    }
}

__global__ static void genScanFilter_and_geq(char *col, int colSize, long tupleNum, struct whereExp * where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = (stringCmp(col+colSize*i, where->content, colSize) >= 0);
        filter[i] &= con;
    }
}

__global__ static void genScanFilter_and_leq(char *col, int colSize, long tupleNum, struct whereExp * where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = (stringCmp(col+colSize*i, where->content, colSize) <= 0);
        filter[i] &= con;
    }
}

__global__ static void genScanFilter_and_soa(char *col, int colSize, int  colType, long tupleNum, struct whereExp * where, int * filter){

    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int rel = where->relation;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        int cmp = 0;
        for(int j=0;j<colSize;j++){
            int pos = j*tupleNum + i; 
            if(col[pos] > where->content[j]){
                cmp = 1;
                break;
            }else if (col[pos] < where->content[j]){
                cmp = -1;
                break;
            }
        }

        if (rel == EQ){
            con = (cmp == 0);
        }else if(rel == LTH){
            con = (cmp <0);
        }else if(rel == GTH){
            con = (cmp >0);
        }else if (rel == LEQ){
            con = (cmp <=0);
        }else if (rel == GEQ){
            con = (cmp >=0);
        }

        filter[i] &= con;
    }
}

__global__ static void genScanFilter_init_int_eq(char *col, long tupleNum, int where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((int*)col)[i] == where; 
        filter[i] = con;
    }
}
__global__ static void genScanFilter_init_float_eq(char *col, long tupleNum, float where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((float*)col)[i] == where; 
        filter[i] = con;
    }
}

__global__ static void genScanFilter_init_int_gth(char *col, long tupleNum, int where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((int*)col)[i] > where; 
        filter[i] = con;
    }
}

__global__ static void genScanFilter_init_float_gth(char *col, long tupleNum, float where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((float*)col)[i] > where; 
        filter[i] = con;
    }
}
__global__ static void genScanFilter_init_int_lth(char *col, long tupleNum, int where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((int*)col)[i] < where; 
        filter[i] = con;
    }
}
__global__ static void genScanFilter_init_float_lth(char *col, long tupleNum, float where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((float*)col)[i] < where; 
        filter[i] = con;
    }
}
__global__ static void genScanFilter_init_int_geq(char *col, long tupleNum, int where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((int*)col)[i] >= where;
        filter[i] = con;
    }
}
__global__ static void genScanFilter_init_float_geq(char *col, long tupleNum, float where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((float*)col)[i] >= where;
        filter[i] = con;
    }
}
__global__ static void genScanFilter_init_int_leq(char *col, long tupleNum, int where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((int*)col)[i] <= where; 
        filter[i] = con;
    }
}
__global__ static void genScanFilter_init_float_leq(char *col, long tupleNum, float where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((float*)col)[i] <= where; 
        filter[i] = con;
    }
}

__global__ static void genScanFilter_and_int_eq(char *col, long tupleNum, int where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((int*)col)[i] == where; 
        filter[i] &= con;
    }
}

__global__ static void genScanFilter_and_float_eq(char *col, long tupleNum, float where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((float*)col)[i] == where; 
        filter[i] &= con;
    }
}

__global__ static void genScanFilter_and_int_geq(char *col, long tupleNum, int where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((int*)col)[i] >= where; 
        filter[i] &= con;
    }
}

__global__ static void genScanFilter_and_float_geq(char *col, long tupleNum, float where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((float*)col)[i] >= where; 
        filter[i] &= con;
    }
}

__global__ static void genScanFilter_and_int_leq(char *col, long tupleNum, int where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((int*)col)[i] <= where; 
        filter[i] &= con;
    }
}

__global__ static void genScanFilter_and_float_leq(char *col, long tupleNum, float where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((float*)col)[i] <= where; 
        filter[i] &= con;
    }
}

__global__ static void genScanFilter_and_int_gth(char *col, long tupleNum, int where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((int*)col)[i] > where; 
        filter[i] &= con;
    }
}

__global__ static void genScanFilter_and_float_gth(char *col, long tupleNum, float where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((float*)col)[i] > where; 
        filter[i] &= con;
    }
}

__global__ static void genScanFilter_and_int_lth(char *col, long tupleNum, int where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((int*)col)[i] < where;
        filter[i] &= con;
    }
}

__global__ static void genScanFilter_and_float_lth(char *col, long tupleNum, float where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((float*)col)[i] < where;
        filter[i] &= con;
    }
}

__global__ static void genScanFilter_init_eq(char *col, int colSize,long tupleNum, struct whereExp * where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = (stringCmp(col+colSize*i, where->content,colSize) == 0);
        filter[i] = con;
    }
}

__global__ static void genScanFilter_init_gth(char *col, int colSize,long tupleNum, struct whereExp * where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = (stringCmp(col+colSize*i, where->content,colSize) > 0);
        filter[i] = con;
    }
}

__global__ static void genScanFilter_init_lth(char *col, int colSize,long tupleNum, struct whereExp * where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = (stringCmp(col+colSize*i, where->content,colSize) < 0);
        filter[i] = con;
    }
}

__global__ static void genScanFilter_init_geq(char *col, int colSize,long tupleNum, struct whereExp * where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = (stringCmp(col+colSize*i, where->content,colSize) >= 0);
        filter[i] = con;
    }
}

__global__ static void genScanFilter_init_leq(char *col, int colSize,long tupleNum, struct whereExp * where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = (stringCmp(col+colSize*i, where->content,colSize) <= 0);
        filter[i] = con;
    }
}

__global__ static void genScanFilter_or_eq(char *col, int colSize, long tupleNum, struct whereExp * where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = (stringCmp(col+colSize *i, where->content, colSize) == 0);
        filter[i] |= con;
    }
}

__global__ static void genScanFilter_or_gth(char *col, int colSize, long tupleNum, struct whereExp * where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = (stringCmp(col+colSize *i, where->content, colSize)> 0);
        filter[i] |= con;
    }
}

__global__ static void genScanFilter_or_lth(char *col, int colSize, long tupleNum, struct whereExp * where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = (stringCmp(col+colSize *i, where->content, colSize) < 0);
        filter[i] |= con;
    }
}

__global__ static void genScanFilter_or_geq(char *col, int colSize, long tupleNum, struct whereExp * where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = (stringCmp(col+colSize *i, where->content, colSize) >= 0);
        filter[i] |= con;
    }
}

__global__ static void genScanFilter_or_leq(char *col, int colSize, long tupleNum, struct whereExp * where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = (stringCmp(col+colSize *i, where->content, colSize) <= 0);
        filter[i] |= con;
    }
}

/*
 * This is only for testing the performance of soa in certain cases.
 */

__global__ static void genScanFilter_or_soa(char *col, int colSize, int  colType, long tupleNum, struct whereExp * where, int * filter){

    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int rel = where->relation;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        int cmp = 0;
        for(int j=0;j<colSize;j++){
            int pos = j*tupleNum + i; 
            if(col[pos] > where->content[j]){
                cmp = 1;
                break;
            }else if (col[pos] < where->content[j]){
                cmp = -1;
                break;
            }
        }

        if (rel == EQ){
            con = (cmp == 0);
        }else if(rel == LTH){
            con = (cmp <0);
        }else if(rel == GTH){
            con = (cmp >0);
        }else if (rel == LEQ){
            con = (cmp <=0);
        }else if (rel == GEQ){
            con = (cmp >=0);
        }

        filter[i] |= con;
    }
}

__global__ static void genScanFilter_or_int_eq(char *col, long tupleNum, int where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((int*)col)[i] == where; 
        filter[i] |= con;
    }
}
__global__ static void genScanFilter_or_float_eq(char *col, long tupleNum, float where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((float*)col)[i] == where; 
        filter[i] |= con;
    }
}

__global__ static void genScanFilter_or_int_gth(char *col, long tupleNum, int where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((int*)col)[i] > where; 
        filter[i] |= con;
    }
}

__global__ static void genScanFilter_or_float_gth(char *col, long tupleNum, float where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((float*)col)[i] > where; 
        filter[i] |= con;
    }
}
__global__ static void genScanFilter_or_int_lth(char *col, long tupleNum, int where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((int*)col)[i] < where; 
        filter[i] |= con;
    }
}
__global__ static void genScanFilter_or_float_lth(char *col, long tupleNum, float where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((float*)col)[i] < where; 
        filter[i] |= con;
    }
}
__global__ static void genScanFilter_or_int_geq(char *col, long tupleNum, int where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((int*)col)[i] >= where;
        filter[i] |= con;
    }
}
__global__ static void genScanFilter_or_float_geq(char *col, long tupleNum, float where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((float*)col)[i] >= where;
        filter[i] |= con;
    }
}
__global__ static void genScanFilter_or_int_leq(char *col, long tupleNum, int where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((int*)col)[i] <= where; 
        filter[i] |= con;
    }
}
__global__ static void genScanFilter_or_float_leq(char *col, long tupleNum, float where, int * filter){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int con;

    for(long i = tid; i<tupleNum;i+=stride){
        con = ((float*)col)[i] <= where; 
        filter[i] |= con;
    }
}

/*
 * countScanNum: count the number of results that each thread generates.
 */

__global__ static void countScanNum(int *filter, long tupleNum, int * count){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int localCount = 0;

    for(long i = tid; i<tupleNum; i += stride){
        localCount += filter[i];
    }

    count[tid] = localCount;

}

/*
 * scan_dict_other: generate the result for dictionary-compressed column.
 */

__global__ static void scan_dict_other(char *col, struct dictHeader * dheader, int byteNum,int colSize, long tupleNum, int *psum, long resultNum, int * filter, char * result){

    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int pos = psum[tid] * colSize;

    for(long i = tid; i<tupleNum; i+= stride){
        if(filter[i] == 1){
            int key = 0;
            memcpy(&key, col + sizeof(struct dictHeader) + i* dheader->bitNum/8, dheader->bitNum/8);
            memcpy(result+pos,&dheader->hash[key],colSize);
            pos += colSize;
        }
    }
}

__global__ static void scan_dict_int(char *col, struct dictHeader * dheader,int byteNum,int colSize, long tupleNum, int *psum, long resultNum, int * filter, char * result){

    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int localCount = psum[tid]; 

    for(long i = tid; i<tupleNum; i+= stride){
        if(filter[i] == 1){
            int key = 0;
            memcpy(&key, col + sizeof(struct dictHeader) + i*byteNum, byteNum);
            ((int *)result)[localCount] = dheader->hash[key];
            localCount ++;
        }
    }
}

/*
 * scan_other: generate scan result for uncompressed column.
 */

__global__ static void scan_other(char *col, int colSize, long tupleNum, int *psum, long resultNum, int * filter, char * result){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int pos = psum[tid]  * colSize;

    for(long i = tid; i<tupleNum;i+=stride){

        if(filter[i] == 1){
            memcpy(result+pos,col+i*colSize,colSize);
            pos += colSize;
        }
    }
}

__global__ static void scan_other_soa(char *col, int colSize, long tupleNum, int *psum, long resultNum, int * filter, char * result){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tNum = psum[tid];

    for(long i = tid; i<tupleNum;i+=stride){

        if(filter[i] == 1){
            for(int j=0;j<colSize;j++){
                long inPos = j*tupleNum + i;
                long outPos = j*resultNum + tNum;
                result[outPos] = col[inPos];
            }
        }
    }
}

__global__ static void scan_int(char *col, int colSize, long tupleNum, int *psum, long resultNum, int * filter, char * result){
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int localCount = psum[tid] ; 

    for(long i = tid; i<tupleNum;i+=stride){

        if(filter[i] == 1){
            ((int*)result)[localCount] = ((int*)col)[i];
            localCount ++;
        }
    }
}

__global__ void static unpack_rle(char * fact, char * rle, long tupleNum, int dNum){

    int offset = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i=offset; i<dNum; i+=stride){

        int fvalue = ((int *)(fact+sizeof(struct rleHeader)))[i];
        int fcount = ((int *)(fact+sizeof(struct rleHeader)))[i + dNum];
        int fpos = ((int *)(fact+sizeof(struct rleHeader)))[i + 2*dNum];

        for(int k=0;k<fcount;k++){
            ((int*)rle)[fpos + k] = fvalue;
        }
    }
}
__global__ void static equal(int * a, int n, unsigned int constC){
    int offset = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i=offset; i<n; i+=stride){    
        a[i] = constC;
    }
}
__global__ void static genScanFilter_int_lth_bit(int * col,int n, unsigned int constC,int * lt, int * eq){
    int offset = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;  

    for(int i=offset; i < n; i+=stride){
        lt[i] = lt[i] | (eq[i] & ~constC & col[i]);
        eq[i] = eq[i] & ~(col[i] ^ constC);
        //printf(" %d %u %u %u\n",i,lt[i],eq[i],col[i]);
    }
}
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", 
            cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

void profilescan(int        *h_a, 
                   int        *h_b, 
                   int        *d, 
                   int *filter,
                   int  n,
                   unsigned int where,
                   char         *desc,
                   unsigned int loopTotal)
{

  dim3 block(256);
  dim3 grid((n + block.x - 1) / block.x);
  float time,stime;
  // events for timing
  cudaEvent_t startEvent, stopEvent; 
  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) ); 
  stime=0;
  int bytes=n * sizeof(int);
  for(int i = 1; i <= loopTotal; i++){  
      checkCuda( cudaEventRecord(startEvent, 0) );
      checkCuda( cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice) );
      // printf("%d\n",    clock());
      genScanFilter_init_int_lth<<<grid,block>>>((char *)d, n,  where, filter);
      checkCuda(cudaThreadSynchronize());
      checkCuda( cudaMemcpy(h_b, filter, bytes, cudaMemcpyDeviceToHost) );
      checkCuda( cudaEventRecord(stopEvent, 0) );
      checkCuda( cudaEventSynchronize(stopEvent) );

      checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
      stime += time;
      //printf("%f\n",stime);
  }
  printf("%f\n" ,bytes * 1e-6/(stime / loopTotal));
  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );
}
void profilebitscan(int        *h_a, 
                   int        *h_b, 
                   int        *d, 
                   int *lt,
                   int *eq,
                   int  n,
                   unsigned int where,
                   char         *desc,
                   unsigned int loopTotal)
{

  dim3 block(256);
  dim3 grid(2048);
  float time,stime;
  // events for timing
  cudaEvent_t startEvent, stopEvent; 
  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) ); 
  stime=0;
  int bytes=n * sizeof(int);
  for(int loop = 1; loop <= loopTotal; loop++){  
      checkCuda( cudaEventRecord(startEvent, 0) );
      checkCuda( cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice) );
      unsigned int c = 0;
      for(int i = 0;i < 32;i++) c += (1u << i);
      equal<<<grid,block>>>(lt, n/32, 0) ;
      equal<<<grid,block>>>(eq, n/32, c) ;
      // printf("%d\n",    clock());
      for(int j = 0; j < 32; ++j){
            int constC = 0;
            for(int k = 0;k < 32;++k) {
            	constC += ((((1U << (31 - j )) & where)>>(31-j))<< k);
               // printf("%u %u\n",j,((((1U << (31 - j )) & where)>>(31-j))<< k));
            }
            genScanFilter_int_lth_bit<<<grid,block>>>(d + j * (n / 32), n / 32,  constC, lt, eq);
            checkCuda(cudaThreadSynchronize());
      }

      checkCuda( cudaMemcpy(h_b, lt, n / 32 * 4, cudaMemcpyDeviceToHost) );
      checkCuda( cudaMemcpy(h_b + n / 32 , eq, n / 32 * 4, cudaMemcpyDeviceToHost) );
      checkCuda( cudaEventRecord(stopEvent, 0) );
      checkCuda( cudaEventSynchronize(stopEvent) );

      checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
      stime += time;
      //printf("%f\n",stime);
  }
  printf("%f\n" ,bytes * 1e-6/(stime / loopTotal));
  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );
}


void profilebitweavscan(int        *h_a, 
                   int        *h_b, 
                   int        *d, 
                   int *lt,
                   int *eq,
                   int  n,
                   unsigned int where,
                   char         *desc,
                   unsigned int loopTotal)
{

  dim3 block(256);
  dim3 grid(2048);
  float time,stime;
  // events for timing
  cudaEvent_t startEvent, stopEvent; 
  checkCuda( cudaEventCreate(&startEvent) );
  checkCuda( cudaEventCreate(&stopEvent) ); 
  stime=0;
  int bytes=n * sizeof(int);
  for(int loop = 1; loop <= loopTotal; loop++){  
      checkCuda( cudaEventRecord(startEvent, 0) );

      unsigned int c = 0;
      for(int i = 0;i < 32;i++) c += (1u << i);
      equal<<<grid,block>>>(lt, n/32, 0) ;
      equal<<<grid,block>>>(eq, n/32, c) ;
      // printf("%d\n",    clock());
      for(int j = 0; j < 32; ++j){
            checkCuda( cudaMemcpy(d + j * (n / 32), h_a + j * (n / 32), n / 32 *4, cudaMemcpyHostToDevice) );
            int constC = 0;
            for(int k = 0;k < 32;++k) {
            	constC += ((((1U << (31 - j )) & where)>>(31-j))<< k);
               // printf("%u %u\n",j,((((1U << (31 - j )) & where)>>(31-j))<< k));
            }
            genScanFilter_int_lth_bit<<<grid,block>>>(d + j * (n / 32) , n / 32,  constC, lt, eq);
            checkCuda(cudaThreadSynchronize());
            
      }

      checkCuda( cudaMemcpy(h_b, lt, n / 32 * 4, cudaMemcpyDeviceToHost) );
      checkCuda( cudaMemcpy(h_b + n / 32 , eq, n / 32 * 4, cudaMemcpyDeviceToHost) );
      checkCuda( cudaEventRecord(stopEvent, 0) );
      checkCuda( cudaEventSynchronize(stopEvent) );

      checkCuda( cudaEventElapsedTime(&time, startEvent, stopEvent) );
      stime += time;
      //printf("%f\n",stime);
  }
  printf("%f\n" ,bytes * 1e-6/(stime / loopTotal));
  checkCuda( cudaEventDestroy(startEvent) );
  checkCuda( cudaEventDestroy(stopEvent) );
}
int main(int argc, char ** argv)
{
  dim3 block(256);
  dim3 grid(2048);
  int inputN;
  sscanf(argv[1],"%d",&inputN);
  unsigned int nElements = inputN;
  const unsigned int bytes = nElements * sizeof(int);

  // host arrays
  int *h_aPageable, *h_bPageable,*h_bitPageable;   
  int *h_aPinned, *h_bPinned;

  // device array
  int *d_a,*lt,*eq;

  // allocate and initialize
  h_aPageable = (int*)malloc(bytes );          
  h_bPageable = (int*)malloc(bytes );
  h_bitPageable =(int *)malloc(bytes );             // host pageable
  checkCuda( cudaMallocHost((void**)&h_aPinned, bytes  ) ); // host pinned
  checkCuda( cudaMallocHost((void**)&h_bPinned, bytes  ) );  
  checkCuda( cudaMalloc((void**)&d_a, bytes  ) );           // device
  checkCuda( cudaMalloc((void**)&lt, bytes ) ); // device return
  checkCuda( cudaMalloc((void**)&eq, bytes  ) );  
  srand(time(0));
  for (int i = 0; i < nElements; ++i) h_aPageable[i] = i;     
  // for (int i = 0; i < nElements; ++i) 
  //   for(int j = 31; j >= 0; --j){
  //       h_bitPageable[i / 32 + (31-j)*(nElements/32)] += (((h_aPageable[i] &(1<<j))>>j)<<(31 - i % 32));
  //   }
  
  memcpy(h_aPinned, h_aPageable, bytes  );

  memset(h_bPageable, 0, bytes);
  memset(h_bPinned, 0, bytes);

  // output device info and transfer size
  cudaDeviceProp prop;

  checkCuda( cudaGetDeviceProperties(&prop, 0) );

  // printf("\nDevice: %s\n", prop.name);
  // if(bytes< 1024){
  //     printf("scan size (B): %d\n", bytes);
  // }else if (bytes < 1024 * 1024)
  // {
  //     printf("scan size (KB): %d\n", bytes / (1024));              
  // }else{
  //     printf("scan size (MB): %d\n", bytes / (1024 * 1024));
  // }


  int constC = nElements/2-1;
  time_t a=clock();
  for (int i = 0; i < nElements; ++i) if(h_aPageable[i] < constC)
        h_bPageable[i] = 1;
  else h_bPageable[i] = 0;
  time_t b=clock();
  double lentime=(double)(b-a)/CLOCKS_PER_SEC;
  printf("%f\n",lentime);
  printf("%f\n",bytes * 1e-9/lentime);
  return;


  // perform  scan eq
 // profilescan(h_aPageable, h_bPageable, d_a, filter, nElements, constC,"Pageable",20);
  //profilescan(h_aPinned, h_bPinned, d_a, filter,nElements, constC,"Pinned",20);
  profilebitscan(h_bitPageable, h_bPageable, d_a, lt, eq, nElements, constC,"Pageable",1);
  // for(int i = 0; i < nElements; i++) printf("%3u ",h_aPageable[i]);printf("\n");
  // for(int i = 0; i < nElements; i++) printf("%3u ",((h_bPageable[i/32] & (1u << (31 - i % 32)))>> (31 - i % 32)));printf("\n");
  // for(int i = 0; i < nElements; i++) printf("%3u ",((h_bPageable[i/32 + nElements/32] & (1u << (31 - i % 32)))>> (31 - i % 32)));printf("\n");
  // for(int i = 0; i < nElements; i++) printf("%3u ",h_bitPageable[i]);printf("\n");
  // cleanup
  cudaFree(lt);
  cudaFree(eq);
  cudaFreeHost(h_aPinned);
  cudaFreeHost(h_bPinned);
  free(h_aPageable); 
}
