 nvcc bitweavingscan+knowstop+smallstore.cu -w
./a.out 1048576 1048576 1024
diff scan.out scan_smallstore.out