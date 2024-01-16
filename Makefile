main:
	clang++ main.cpp Position.cpp -I./libtorch/lib -Llibtorch -O3 -Wno-narrowing -flto -mbmi -o ./build/out -std=c++17
	./build/out

table:
	clang++ tableGenerator.cpp Position.cpp -o ./build/tables -mbmi -O2 -std=c++20
	./build/tables
