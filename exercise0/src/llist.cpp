/*
 * Copyright 2011 by martinp.dev@googlemail.com
 */

#include <stdio.h>
#include <stdlib.h>
#include "llist.h"

ListElement* addStart(ListElement *list, int val)
{
    ListElement *element = (ListElement *) malloc(sizeof(ListElement));
    element->value = val;
    element->next = list;

	return element;
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

ListElement* add(ListElement *list, int pos, int val)
{
    if(pos < 0)
    {
        return list;
    }


    //Start a new List
    if(list == NULL)
    {
        list = (ListElement *) malloc(sizeof(ListElement));
        list->value = val;
        list->next = NULL;
        return list;
    }

    ListElement *element;

    //Insert at the begining
    if(pos == 0)
    {
        element = (ListElement *) malloc(sizeof(ListElement));
        element->value = val;
        element->next = list;
        return element;
    }

    //Iterate pos-1 times forward in the list
    int i;
    element = list;
    for(i = 0; i < pos-1; i++)
    {
        if(element->next == NULL && i < pos-2)
        {
            //pos is after the end of the list, nothing to do
            return list;
        }
        element = element->next;
    }

    ListElement *newelement = (ListElement *) malloc(sizeof(ListElement));
    newelement->value = val;
    newelement->next = element->next;
    element->next = newelement;
    return list;
}

ListElement* removeElements(ListElement *list, int value)
{
    ListElement *last = NULL;
    ListElement *element = list;

    while(element != NULL)
    {
        if(element->value == value)
        {
            if(last == NULL)
            {
                //remove the first element of the list
                ListElement *toremove = element;
                element = element->next;
                list = element;
                free(toremove);
            }
            else
            {
                //remove an element
                last->next = element->next;
                ListElement *toremove = element;
                element = element->next;
                free(toremove);
            }
        }
        else
        {
            //skip an element
            last = element;
            element = element->next;
        }
    }
    return list;
}

ListElement* removeElement(ListElement *list, int pos)
{
    if(pos < 0)
    {
        return list;
    }

    //List is empty
    if(list == NULL)
    {
        return list;
    }

    ListElement *element = list;

    //Insert at the begining
    if(pos == 0)
    {
        list = element->next;
        free(element);
        return list;
    }

    //Iterate pos-1 times forward in the list
    int i;
    element = list;
    for(i = 0; i < pos-1; i++)
    {
        if(element->next == NULL && i < pos-2)
        {
            //pos is after the end of the list, nothing to do
            return list;
        }
        element = element->next;
    }
    ListElement *toremove = element->next;
    element->next = element->next->next;
    free(toremove);
    return list;
}

ListElement* reverse(ListElement *list)
{
    if(list == NULL)
    {
        return NULL;
    }
    if(list->next == NULL)
    {
        return list;
    }
    else
    {
        ListElement *nlist = reverse(list->next);
        list->next->next = list;
        list->next = NULL;
        return nlist;
    }
}

ListElement* getElement(ListElement* list, int pos)
{
    while(list != NULL && pos >= 0)
    {
        if(pos == 0)
        {
            return list;
        }
        pos--;
        list = list->next;
    }
    return NULL;
}

void print(ListElement *list)
{
    printf("List:");
    if(list != NULL)
    {
        printf(" %d", list->value);
        list = list->next;
    }
    while(list != NULL)
    {
        printf(", %d", list->value);
        list = list->next;
    }
    printf("\n");
}

void freeList(ListElement *list)
{
    if(list == NULL)
    {
        return;
    }
    freeList(list->next);
    free(list);
}

int length(ListElement* list)
{
    int length = 0;
    while(list != NULL)
    {
        length++;
        list = list->next;
    }
    return length;
}