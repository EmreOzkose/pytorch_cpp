// https://github.com/Dayof/opencv-socket/blob/master/server.cpp
// https://github.com/Dayof/opencv-socket/blob/master/client.cpp

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <iostream>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>

#include "opencv2/core/core.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define SERVER_URL "127.0.0.1"
#define PORT 7200

#define IM_WIDTH         640
#define IM_HEIGHT        480

#define FRAME_WIDTH         640
#define FRAME_HEIGHT        480


void error(const char *msg)
{
    perror(msg);
    exit(0);
}

