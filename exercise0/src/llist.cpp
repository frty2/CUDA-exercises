#include <stdio.h>
#include <stdlib.h>
#include "llist.h"

ListElement* addStart(ListElement *list, int val)
{
	ListElement *element = (ListElement *) malloc(sizeof(ListElement));
	element->value = val;
	element->next = list;
	
	list = element;
	
	return list;
}

ListElement* addEnd(ListElement *list, int val)
{
	ListElement *element;
	
	if(list == NULL)
	{
		element = (ListElement *) malloc(sizeof(ListElement));
		element->value = val;
		element->next = NULL;
		
		list = element;
		return list;
	}
	element = list;
	while(element->next != NULL)
	{
		element = element->next;
	}
	element->next = (ListElement *) malloc(sizeof(ListElement));
	element->next->value = val;
	element->next->next = NULL;
	
	return list;
}

ListElement * add(ListElement *list, int pos, int val)
{
	ListElement *element = list;
	
	
	
	int i;
	
	for(i = 0;i < pos-1 && element != NULL && element->next != NULL;++i)
	{
		element = element->next;
	}
	
	ListElement *newelement = (ListElement *) malloc(sizeof(ListElement));
	newelement->value = val;
	
	if(element == NULL)
	{
		return newelement;
	}
	else
	{
		newelement->next = element->next;
		element->next = newelement;
		return list;
	}
}

void removeElements(ListElement *list, int value)
{
	
}

void removeElement(ListElement *list, int pos)
{
	
}

void print(ListElement *list)
{
	ListElement *element = list;
	printf("List:\n");
	while(element != NULL)
	{
		printf("List Element: %d\n", element->value);
		element = element->next;
	}
	printf("\n\n\n");
}

void reverse()
{
	
}