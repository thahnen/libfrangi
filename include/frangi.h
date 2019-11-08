#include <opencv2/opencv.hpp>

#define DEFAULT_SIGMA_START 3
#define DEFAULT_SIGMA_END 7
#define DEFAULT_SIGMA_STEP 1
#define DEFAULT_BETA_ONE 1.6
#define DEFAULT_BETA_TWO 0.08
#define DEFAULT_BLACKWHITE true


/// Frangi filter options
typedef struct{
	// vessel scales
	int sigma_start;
	int sigma_end;
	int sigma_step;
    float BetaOne;      // correction constant: suppression of blob-like structures
	float BetaTwo;      // correction constant: background suppression
	bool BlackWhite;    // detection/ enhancement of black if true, otherwise white structures
} frangi2d_opts_t;


/**
 *  Applys a full frangi filter to the source image using provided options
 *
 *  @param src              the source image
 *  @param opts             the options to use
 *  @param vessel           the vessel enhanced image, values equal maximum for all scales
 *  @param scale            scale on which maximum intensity of every pixel is found
 *  @param angle            vessel angles (from minor eigenvector)
 */
void frangi2d(const cv::Mat& src, frangi2d_opts_t opts, cv::Mat& vessel, cv::Mat& scale, cv::Mat& angle);


/// Helper functions from here!

/**
 *  Runs a 2d hessian filter on the source image using sigma
 *
 *                          | Dxx   Dxy |
 *  Matrix looks like that: |           |
 *                          | Dxy   Dyy |
 *
 *  @param src              the source image
 *  @param sigma            sigma of gaussian kernel used
 *  @param Dxx              second derivative (output)
 *  @param Dxy              second derivative (output)
 *  @param Dyy              second derivative (output)
 */
void frangi2d_hessian(const cv::Mat& src, float sigma, cv::Mat& Dxx, cv::Mat& Dxy, cv::Mat& Dyy);


/**
 *  Sets given options to the default values
 *
 *  @param opts             reference to existing options
 */
void frangi2d_createopts(frangi2d_opts_t& opts);


/**
 *  Estimates eigenvalues from Dxx, Dxy, Dyy to the results lambda1, lambda2, Ix, Iy
 *
 *                          | Dxx   Dxy |
 *  Matrix looks like that: |           |
 *                          | Dxy   Dyy |
 *
 *  @param Dxx              second derivative (input)
 *  @param Dxy              second derivative (input)
 *  @param Dyy              second derivative (input)
 *  @param lambda1          eigen value
 *  @param lambda2          eigen value
 *  @param Ix               x direction of structure
 *  @param Iy               y direction of structure
 */
void frangi2_eig2image(const cv::Mat& Dxx, const cv::Mat& Dxy, const cv::Mat& Dyy, cv::Mat& lambda1, cv::Mat& lambda2,
                        cv::Mat& Ix, cv::Mat& Iy);
