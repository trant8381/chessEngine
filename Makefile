main:
	clang++ main.cpp Position.cpp -Wno-narrowing -flto -mbmi -o ./build/out -std=c++17 -I/usr/include/torch/csrc/api/include/torch/torch.h
	./build/out
table:
	clang++ tableGenerator.cpp Position.cpp -o ./build/tables -mbmi -O2 -std=c++20
	./build/tables
