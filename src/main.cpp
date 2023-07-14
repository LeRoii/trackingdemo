#include <unistd.h>
#include <signal.h>
#include <opencv2/opencv.hpp>
// #include <opencv2/videoio.hpp>
#include <stdio.h>
#include "sdireader.h"
#include "nvrender.h"
#include "idetector.h"
#include <thread>
#include <termios.h>
#include <fcntl.h>
#include "common.h"
#include "tracker.h"

bool quit = false;

const int DISP_IMG_W = 1280;
const int DISP_IMG_H = 720;

void fusion(std::vector<StObject>& objs);

static void signal_handle(int signum)
{
	quit = true;
	// _xdma_reader_ch0.xdma_close();
}


int DispFlag = 0;
void process_keyboard_events()
{
    struct termios old_tio, new_tio;
    unsigned char c;
    int tty_fd;

    // 打开标准输入设备（键盘）设备文件
    tty_fd = open("/dev/tty", O_RDONLY | O_NONBLOCK);
    if (tty_fd == -1)
    {
        std::cerr << "无法打开标准输入设备文件" << std::endl;
        quit = 1;
        return;
    }

    // 保存并修改终端设置  
    tcgetattr(tty_fd, &old_tio);
    new_tio = old_tio;
    new_tio.c_lflag &= (~ICANON & ~ECHO);
    tcsetattr(tty_fd, TCSANOW, &new_tio);

    while (!quit)
    {
        // 读取一个字符
        if (read(tty_fd, &c, 1) > 0)
        {
            std::cout <<c<< std::endl;
            // 处理按键事件
            switch (c)
            {
            case 'q': 
                DispFlag = 0;
                std::cout << "按键q已按下" << std::endl;
                break;
            case 'w': 
                DispFlag = 1;
                std::cout << "按键w已按下" << std::endl;
                break;
            case 'e':
                DispFlag = 2;
				std::cout << "按键e已按下" << std::endl;
                break;
            case 'r':
                break;
            case 'f':
                break;
            }
        }

        // 程序暂停一段时间，避免CPU占用率过高
        usleep(1000);
    }

    // 还原终端设置
    tcsetattr(tty_fd, TCSANOW, &old_tio);
    close(tty_fd);
}



int main(int argc, char* argv[])
{

	struct sigaction sig_action;
	sig_action.sa_handler = signal_handle;
	sigemptyset(&sig_action.sa_mask);
	sig_action.sa_flags = 0;
	sigaction(SIGINT, &sig_action, NULL);

	nvrenderCfg renderCfg{DISP_IMG_W,DISP_IMG_H,DISP_IMG_W,DISP_IMG_H,0,0,0};
	nvrender *render;

    cv::VideoCapture cap("/home/nx/code/testfus/test.mp4");

	// render = new nvrender(renderCfg);
	// idetector* vis_detector = new idetector("/home/nxs/model/yolov5s.engine");
    idetector* vis_detector = new idetector("/home/nx/model/vis.engine");
	// cv::VideoCapture IRCamera("rtspsrc location=rtsp://admin:admin123456@192.168.1.107 latency=0 ! rtph264depay ! h264parse ! omxh264dec ! videoconvert ! appsink max-buffers=1 drop=true sync=false", cv::CAP_GSTREAMER);
    // cv::VideoCapture IRCamera1("rtspsrc location=rtsp://admin:admin123456@192.168.1.108 latency=0 ! rtph264depay ! h264parse ! omxh264dec ! videoconvert ! appsink max-buffers=1 drop=true sync=false", cv::CAP_GSTREAMER);


	cv::Mat frame0, frame1;
	cv::Mat irImg, irImg1;
	cv::Mat ret;
	cv::Mat out;
	cv::Mat DispImg;

	// Sdireader_Init("/etc/jetsoncfg/NXConfig.ini");

	std::thread keyboard_event_thread(process_keyboard_events);


	sleep(5);

    int framecnt = 0;

    CTracker *tracker = new CTracker();

	while(!quit)
	{
		// if(DispFlag == 1)
		// {
		// 	IRCamera >> DispImg;
		// }
		// else if(DispFlag == 2)
		// {
		// 	IRCamera1 >> DispImg;
		// }
		// else
		// {
		// 	Sdireader_GetFrame(DispImg, frame1);
		// }

        cap >> DispImg;

        if(DispImg.empty())
            return 0 ;
		std::vector<bbox_t> boxs;
        std::vector<StObject> objs;

        auto DetImg = DispImg.clone();

		vis_detector->process(DetImg,ret, boxs);

        printf("framecnt:%d:\n",framecnt++);
        for(int i=0;i<boxs.size();i++)
        {
            StObject obj;
            obj.longitude = obj.x = boxs[i].x;
            obj.latitude = obj.y = boxs[i].y;
            obj.w = boxs[i].w;
            obj.h = boxs[i].h;
            obj.clsId = boxs[i].obj_id;
            objs.push_back(obj);
            printf("obj:%d, x:%d,y:%d,w:%d,h:%d\n", i, boxs[i].x,boxs[i].y,boxs[i].w,boxs[i].h);
        }


        tracker->update(objs);

        std::vector<StObject> tracks;
        tracker->GetTracks(tracks);

        for(auto& track:tracks)
        {
            cv::Point topLeft(track.x, track.y);
            cv::Point bottomRight(track.x + track.w, track.y + track.h);
            cv::rectangle(DispImg, topLeft, bottomRight, cv::Scalar(255,0,0), 2);
            cv::putText(DispImg, std::to_string(track.objId), cv::Point(track.x, track.y + 4), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar::all(0xFF), 2);

        }

		cv::resize(DispImg, DispImg, cv::Size(DISP_IMG_W,DISP_IMG_H));

		// render->render(DispImg);

		cv::imshow("1", DispImg);
		cv::imshow("2", DetImg);
		cv::waitKey(0);
	}

	return 0;
}

