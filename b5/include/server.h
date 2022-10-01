#include <utils.h>


struct SERVER
{
    int sockfd;
    int newsockfd;
    int n;
    int bytes=0;

    socklen_t clilen;
    struct sockaddr_in serv_addr, cli_addr;
    
    SERVER(){
        sockfd=socket(AF_INET, SOCK_STREAM, 0);
        if( sockfd < 0 ) error("ERROR opening socket");

        bzero((char*)&serv_addr, sizeof(serv_addr));

        serv_addr.sin_family=AF_INET;
        serv_addr.sin_addr.s_addr=INADDR_ANY;
        serv_addr.sin_port=htons(PORT);
        
        if(bind(sockfd, (struct sockaddr *) &serv_addr, sizeof(serv_addr))<0) error("ERROR on binding");
    }

    cv::Mat receive_img(bool show_image = false){
        listen(sockfd, 10);
        clilen=sizeof(cli_addr);

        newsockfd=accept(sockfd, (struct sockaddr *) &cli_addr, &clilen);
        if(newsockfd<0) error("ERROR on accept");

        uchar sock[3];

        cv::Mat img = cv::Mat::zeros(IM_HEIGHT, IM_WIDTH, CV_8UC3);

        int imgSize = img.total()*img.elemSize();
        uchar sockData[imgSize];

        for(int i=0;i<imgSize;i+=bytes)
            if ((bytes=recv(newsockfd, sockData+i, imgSize-i,0))==-1) error("recv failed");

        int ptr=0;

        for(int i=0;i<img.rows;++i)
            for(int j=0;j<img.cols;++j){
                img.at<cv::Vec3b>(i,j) = cv::Vec3b(sockData[ptr+0],sockData[ptr+1],sockData[ptr+2]);
                ptr=ptr+3;
            }

        if (show_image){
            cv::namedWindow( "Server", cv::WINDOW_AUTOSIZE );// Create a window for display.
            cv::imshow( "Server", img );
            
            cv::waitKey(0);
            cv::destroyWindow("Server");
        }

        return img;
    }
};