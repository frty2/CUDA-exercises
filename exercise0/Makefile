TARGET := listtest

OBJDIR := obj
TARGETDIR := bin
SRCDIR := src

CXX := g++ -g -O2
LNK := g++

OBJ := $(OBJDIR)/llist.o $(OBJDIR)/main.o

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) -c $< -o $@

all:
	mkdir -p $(OBJDIR)/
	mkdir -p $(TARGETDIR)/
	make test

test: $(OBJ)
	$(LNK) -o $(TARGETDIR)/$(TARGET) $^

clean:
	rm -fR $(OBJDIR)/
	rm -fR $(TARGETDIR)/

start:
	./$(TARGETDIR)/$(TARGET)
