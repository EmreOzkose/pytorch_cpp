#include <utils.h>


struct CLIENT{
	struct sockaddr_in serv_addr;
	struct hostent *server;
	int sockfd;

	CLIENT(){
		server = gethostbyname(SERVER_URL);
		if (server == NULL) {
			fprintf(stderr,"ERROR, no such host\n");
			exit(0);
		}

		sockfd = socket(AF_INET, SOCK_STREAM, 0);
		if (sockfd < 0) error("ERROR opening socket");

		bzero((char *) &serv_addr, sizeof(serv_addr));
		serv_addr.sin_family = AF_INET;
		bcopy((char *)server->h_addr,
			(char *)&serv_addr.sin_addr.s_addr,
			server->h_length);
		serv_addr.sin_port = htons(PORT);

		if (connect(sockfd,(struct sockaddr *) &serv_addr,sizeof(serv_addr)) < 0)
			error("ERROR connecting");
	}

    void send_image(cv::Mat image){
        auto imgSize=image.total()*image.elemSize();

        auto n = send(sockfd, image.data, imgSize, 0);
        if (n < 0) error("ERROR writing to socket");
    }
};
