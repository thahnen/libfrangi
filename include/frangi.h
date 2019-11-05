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
	
	// BetaOne: suppression of blob-like structures.
    float BetaOne;

	// BetaTwo: background suppression. (See Frangi1998...)
	float BetaTwo;

	bool BlackWhite; //enhance black structures if true, otherwise enhance white structures
} frangi2d_opts_t;


/**
 *  Applys a full frangi filter to the source image using provided options
 *
 *  @param src              the source image
 *  @param vessel           result storage for vesselness
 *  @param scale            result storage for scale
 *  @param angle            result storage for vessel angle
 *  @param opts             the options to use
 */
void frangi2d(const cv::Mat& src, cv::Mat& vessel, cv::Mat& scale, cv::Mat& angle, frangi2d_opts_t opts);


/// Helper functions from here!

/**
 *  Runs a 2d hessian filter on the source image using sigma
 *
 *  @param src              the source image
 *  @param Dxx              result storage
 *  @param Dxy              result storage
 *  @param Dyy              result storage
 *  @param sigma            yet unknown parameter
 */
void frangi2d_hessian(const cv::Mat& src, cv::Mat& Dxx, cv::Mat& Dxy, cv::Mat& Dyy, float sigma);


/**
 *  Sets given options to the default values
 *
 *  @param opts             pointer to existing options
 */
void frangi2d_createopts(frangi2d_opts_t* opts);


/**
 *  Estimates eigenvalues from Dxx, Dxy, Dyy to the results lambda1, lambda2, Ix, Iy
 *
 *  @param Dxx              source for estimation
 *  @param Dxy              source for estimation
 *  @param Dyy              source for estimation
 *  @param lambda1          result storage
 *  @param lambda2          result storage
 *  @param Ix               result storage
 *  @param Iy               result storage
 */
void frangi2_eig2image(const cv::Mat& Dxx, const cv::Mat& Dxy, const cv::Mat& Dyy, cv::Mat& lambda1, cv::Mat& lambda2, cv::Mat& Ix, cv::Mat& Iy);
