#include <stdio.h>
#include "llist.h"

ListElement* createOneElementedList(int value)
{
	
}

ListElement* generateNRandomList(int n)
{
	
}

ListElement* AddAtBeginAndEnd(ListElement* list)
{
	
}

ListElement* appendAtMiddle(ListElement* list)
{
	
}

ListElement* removeAllEqual(ListElement* list, int value)
{
	
}

ListElement* removeSecond(ListElement* list)
{
	
}

ListElement* reverseOrder(ListElement* list)
{
	
}


int main(int argc, char **argv)
{
	ListElement *list = NULL;
	list = addEnd(list, 42);
	list = addEnd(list, 5);
	list = addStart(list, 7);
	list = add(list, 1, 1337);
	list = add(list, 3, 7);
	list = add(list, 3, 7);
	list = addStart(list, 2);
	list = addEnd(list,  42);
	print(list);
	list = reverse(list);
	print(list);
	list = removeElements(list, 42);
	print(list);
	list = removeElement(list, 0);
	print(list);
	
	freeList(list);
	return 0;
}