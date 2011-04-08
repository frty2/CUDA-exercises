#include <stdio.h>
#include "llist.h"

int main(int argc, char **argv)
{
	ListElement *list;
	list = addEnd(list, 42);
	list = addEnd(list, 5);
	list = addStart(list, 7);
	list = add(list, 0, 1337);
	print(list);
	
	return 0;
}