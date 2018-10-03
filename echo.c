/*
 * For this project, my goal was to write a program that takes words as command line arguments and prints each word twice, once as it is and once all-capitalized, separated by  space.
 */

/*
 * echo.c
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

void string_upper(char *string)
{
    int i;
    for(i = 0; string[i] != '\0'; i++)
    {
        if(string[i] >= 'a' && string[i] <= 'z')
            string[i] = 'A' + string[i] - 'a';
    }
}

char** duplicateArgs(int argc, char **argv)
{
    int length;
    char **newargs = (char**) malloc((argc + 1) * sizeof(char*));
    char *input;
    int i; 

    for(i = 1; i < argc; i++)
    {
        length = strlen(argv[i]);
        input = (char *) malloc((length + 1) * sizeof(char));
        strcpy(input, argv[i]);
        string_upper(input);
        newargs[i] = input;
    }

    newargs[i] = NULL; 
    return newargs;
}

void freeDuplicatedArgs(char **copy)
{
    int i = 0;
    // printf("free space allocated to copy.\n");

    for(i = 0; copy[i] != NULL; i++)
    {
        free(copy[i]);
        // printf("copy[%d] is now free.]\n", i);
    }

    free(copy);
}

/*
 * DO NOT MODIFY main().
 */
int main(int argc, char **argv)
{
    if (argc <= 1)
        return 1;

    char **copy = duplicateArgs(argc, argv);
    char **p = copy;

    argv++;
    p++;
    while (*argv) {
        printf("%s %s\n", *argv++, *p++);
    }

    freeDuplicatedArgs(copy);

    return 0;
}

