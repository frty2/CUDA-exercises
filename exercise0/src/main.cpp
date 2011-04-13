/*
 * Copyright 2011 by martinp.dev@googlemail.com
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "llist.h"

ListElement* createOneElementedList(ListElement *list, int value)
{
    return addStart(list, value);
}

ListElement* generateNRandomList(int n)
{
    srand ( time(NULL) );

    ListElement *list = NULL;
    int i;
    for(i = 0; i < n; i++)
    {
        int random = rand() % 999 + 1;
        list = addStart(list, random);
    }
    return list;
}

ListElement* AddAtBeginAndEnd(ListElement* list)
{
    list = addStart(list, 4);
    return addEnd(list, 2);
}

ListElement* appendAtMiddle(ListElement* list)
{
    int middle = length(list) / 2;
    printf("Element in th middle: %d\n", getElement(list, middle)->value);

    return add(list, middle + 1, 300);
}

ListElement* removeAllEqual(ListElement* list)
{
    int middle = length(list) / 2;
    int toremove = getElement(list, middle)->value;

    return removeElements(list, toremove);
}

ListElement* removeSecond(ListElement* list)
{
    return removeElement(list, 1);
}

ListElement* reverseOrder(ListElement* list)
{
    return reverse(list);
}


int main(int argc, char **argv)
{
    int n = 10;

    if(argc == 2)
    {
        n = atoi(argv[1]);
    }
    else
    {
        printf("Use './listtest n'. Setting n = %d as default\n", n);
    }

    ListElement *list;

    list = createOneElementedList(list, 42);
    print(list);
    freeList(list);

    list = generateNRandomList(n);
    print(list);

    list = AddAtBeginAndEnd(list);
    print(list);

    list = appendAtMiddle(list);
    print(list);

    list = removeAllEqual(list);
    print(list);

    list = removeSecond(list);
    print(list);

    list = reverseOrder(list);
    print(list);

    freeList(list);
    return 0;
}