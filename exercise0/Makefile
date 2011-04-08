TARGET := exercise0
CXX := gcc -g -O2
LNK := gcc
OBJ := obj/llist.o obj/main.o


obj/%.o: src/%.cpp
	$(CXX) -c $< -o $@

all:
	make test

test: $(OBJ)
	$(LNK) -o bin/$(TARGET) $^

clean:
	rm -f obj/*.o