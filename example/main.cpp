#include <iostream>
#include <string>
#include <vector>

#include "frangi.h"


int main(int argc, char* argv[]){
    std::vector<std::string> images = {
            "../example/assets/frame-1.png", "../example/assets/frame-2.png", "../example/assets/frame-3.png",
            "../example/assets/frame-4.png", "../example/assets/frame-5.png", "../example/assets/frame-6.png",
            "../example/assets/frame-7.png", "../example/assets/frame-8.png", "../example/assets/frame-9.png",
            "../example/assets/frame-10.png", "../example/assets/frame-11.png", "../example/assets/frame-12.png",
            "../example/assets/frame-13.png", "../example/assets/frame-14.png", "../example/assets/frame-15.png",
            "../example/assets/frame-16.png", "../example/assets/frame-17.png", "../example/assets/frame-18.png",
            "../example/assets/frame-19.png", "../example/assets/frame-20.png",
    };

    for (const auto& path : images) {
        // Set default frangi options
        frangi2d_opts_t opts;
        frangi2d_createopts(opts);

        // Read image from file
        cv::Mat input = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (!input.data) {
            std::cerr << "Image not found or could not be loaded: " << path << std::endl;
            return 1;
        }

        // Run frangi on given (now colered) image
        cv::Mat working, vesselness, scale, angles;
        input.convertTo(working, CV_32FC3);
        frangi2d(working, opts, vesselness, scale, angles);

        // Save output image to file!
        imwrite(path + ".out", vesselness * 255);
    }

	return 0;
}
