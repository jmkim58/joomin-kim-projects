/*
 * For this project, my goal was to write a program called mdb-lookup-server.c that lets the user see what is inside the database and search for a particular name or message. The program communicates with the client via a socket and reads and writes using the socket.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>

#include "mylist.h"
#include "mdb.h"

#define KeyMax 5 

static void die(const char *s) { perror(s); exit(1); }

int main(int argc, char **argv)
{
    if (signal(SIGPIPE, SIG_IGN) == SIG_ERR)
        die("signal() failed");
    if (argc != 3) {
        fprintf(stderr, "usage: %s <database> <server-port>\n", argv[0]);
        exit(1);
    }

    unsigned short port = atoi(argv[2]);
    char to_send[512];
    const char *filename = argv[1];
    
    // Create a listening socket

    int servsock;
    if ((servsock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
        die("socket failed");

    // Construct local address structure
    
    struct sockaddr_in servaddr;
    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = htonl(INADDR_ANY); // any network interfaec
    servaddr.sin_port = htons(port); 

    // Bind to local address
    
    if (bind(servsock, (struct sockaddr *) &servaddr, sizeof(servaddr)) < 0)
        die("bind failed");

    // Start listening for incoming connections

    if (listen(servsock, 5 /* queue size for connection requests */ ) < 0)
        die("listen failed");

    int clntsock;
    socklen_t clntlen;
    struct sockaddr_in clntaddr;

    while (1) {

        clntlen = sizeof(clntaddr); // in-out parameter

        if ((clntsock = accept(servsock,
                        (struct sockaddr *) &clntaddr, &clntlen)) < 0)
            die("accept failed");


        fprintf(stderr, "client IP: %s\n", inet_ntoa(clntaddr.sin_addr));
        FILE *input = fdopen(clntsock, "r");

        char *filename = argv[1];
        FILE *fp = fopen(filename, "rb");
        if (fp == NULL)
        	die(filename);

        struct List list;
        initList(&list);

	while (fread(&r, sizeof(r), 1, fp) == 1) {

		struct MdbRec *rec = (struct MdbRec *)malloc(sizeof(r));
		if (!rec)
			die("malloc failed");
		memcpy(rec, &r, sizeof(r));

		node = addAfter(&list, node, rec);
		if (node == NULL)
			die("addAfter failed");

	}

	if (ferror(fp))
		die("fread failed");

	fclose(fp);


	// look-up loop 

	char line[1000];
        char key[KeyMax + 1];
        while (fgets(line, sizeof(line), input) != NULL) {

                    strncpy(key, line, sizeof(key) - 1);
                    key[sizeof(key) - 1] = '\0';

                    // Remove new line if present
                    size_t last = strlen(key) - 1;
                    if (key[last] == '\n')
                        key[last] = '\0';

                    struct Node *node = list.head;
                    int recNo = 1;
                    while (node) {
                       struct MdbRec *rec = (struct MdbRec *)node->data;
                       if (strstr(rec->name, key) || strstr(rec->msg, key)) {
                           snprintf(to_send, sizeof(to_send), "%4d: {%s} said {%s}\n", recNo, rec->name, rec->msg);
                           send(clntsock, to_send, strlen(to_send) + 1, 0);
                       }

                       node = node->next;
                       recNo++;
                    }

		size = sprintf(buf, "\n");
		if (send(clntsock, buf, size, 0) != size)
			    perror("send content failed");
       	}

       	if (ferror(input)) {
       	perror("fgets failed to read from input");
	}

	traverseList(&list, &free);
	removeAllNodes(&list);

	fclose(input);

	fprintf(stderr, "connection termianted from: %s\n", 
				inet_ntoa(clntaddr.sin_addr));

    }
}

