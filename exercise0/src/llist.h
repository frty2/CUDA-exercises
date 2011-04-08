struct ListElement 
{
	int value;
	struct ListElement *next;
};

typedef struct ListElement ListElement;
	
	
	ListElement* addStart(ListElement *list, int val);
	
	ListElement* addEnd(ListElement *list, int value);
	
	ListElement* add(ListElement *list, int pos, int value);
	
	void removeElements(ListElement *list, int value);
	
	void removeElement(ListElement *list, int pos);
	
	void print(ListElement *list);
	
	void reverse();