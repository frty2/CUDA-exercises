struct ListElement 
{
	int value;
	struct ListElement *next;
};

typedef struct ListElement ListElement;


ListElement* addStart(ListElement *list, int val);

ListElement* addEnd(ListElement *list, int value);

ListElement* add(ListElement *list, int pos, int value);

ListElement* removeElements(ListElement *list, int value);

ListElement* removeElement(ListElement *list, int pos);

ListElement* reverse(ListElement *list);

ListElement* getElement(int pos);

void print(ListElement *list);

void freeList(ListElement *list);

int length(ListElement *list);