/*
 * For this project, my goal was to write a program that dynamically allocates an array o integers assuming that the user will input a positive integer. The elements of the array will be filled using random() function. After filling the array with random numbers, the program makes a copy of the array, and sorts the new array in ascending order. Then, pthe program makes a second copy of the original array and sorts it in descending array. Finaly, the program prints out all three arrays (used malloc() library function to separately allocate the three arrays.
 */

/*
 * sort.c
 */

#include <stdio.h> 
#include <stdlib.h>
#include <time.h>
#include <malloc.h>

// calling function given in instructions 

void sort_integer_array(int *begin, int *end, int ascending) 
{
    if (ascending == 1)
    {

        int *start = begin;
        int *endLess = (end - 1); 
        int *currentElem = begin;
        int *elemToSort;
        while (currentElem <= endLess)
        {
            elemToSort = currentElem;

            while (elemToSort <= endLess)
            {
                if (*currentElem > *elemToSort)
                {
                    *currentElem ^= *elemToSort;
                    *elemToSort ^= *currentElem;
                    *currentElem ^= *elemToSort;
                }
                elemToSort++;
            }
            currentElem++;
        }   

        while (start < end)
        {
            printf("%d ", *start++);
        }

     }
     else if (ascending == 0)
     { 
            int *start = begin;
            int *endLess = (end - 1);
            int *currentElem = begin;
            int *elemToSort;
            while (currentElem <= endLess)
            {
                elemToSort = currentElem;

                while (elemToSort <= endLess)
                {
                    if (*currentElem < *elemToSort)
                    {
                        *currentElem ^= *elemToSort;
                        *elemToSort ^= *currentElem;
                        *currentElem ^= *elemToSort;
                    }
                    elemToSort++;
                }
                currentElem++;
            }

            while (start < end)
            {
                printf("%d ", *start++);
            }
      }
}

int main()
{
   int size;
   int *original;

   printf("Enter size of array: "); // ask for user input
   scanf("%d", &size); 

   original = (int*) malloc(size * sizeof(int));
   if (original == NULL)
   { 
       perror("malloc returned NULL");
       exit(1);
   }
   else
   {
       int i;
       printf("\noriginal: ");

       srand((unsigned)time(0));
       for (i = 0; i < size; i++)
       {
           original[i] = rand() % 99; 
           printf("%d ", original[i]);
       }
   }

   // sorting in ascending order 
   int *ascending = malloc(size * sizeof(int));
   if (ascending == NULL)
   {
       perror("malloc returned NULL");
       exit(1);
   }
   else
   {
       int i = 0;
       printf("\nascending: ");

       for (i = 0; i < size; i++)
       {
           ascending[i] = original[i];
       }

       sort_integer_array(&ascending[0], &ascending[size], 1);

   } 

   // sorting in descending order
   int *descending = malloc(size * sizeof(int));
   if (descending == NULL)
   {
       perror("malloc returned NULL");
       exit(1);
   }
   else
   {
       int i = 0;
       printf("\ndescending: ");

       for (i = 0; i < size; i++)
       {
           descending[i] = original[i];
       }

       sort_integer_array(&descending[0], &descending[size], 0);

   }

   // free memory
   free(original);
   // printf("\n\nfreed memory, allocated to array called original\n");
   free(ascending);
   // printf("freed memory, allocated to array called ascending\n");
   free(descending);
   // printf("freed memory, allocated to array called descending\n");
   printf("\n");
   return 0;
}


/*
 * Makefile
 */

CC = gcc
CLFAGS = -g -Wall
LDFLAGS = -g
ISORT = sort

.PHONY: default
default: all run

$(SORT): sort.o
	gcc $(LDFLAGS) sort.o -o $(SORT)

sort.o: sort.c
	gcc $(CFLAGS) -c -o sort.o sort.c

.PHONY: all
all: clean sort

.PHONY: clean
clean: 
	rm -f *.o $(SORT)


.PHONY: run
run: 
	./$(SORT)

