#include <iostream>

#ifndef NO_OMP
#   include <omp.h>
#endif

#include "frangi.h"


/// Applys a full frangi filter to the source image using provided options
void frangi2d(const cv::Mat& src, frangi2d_opts_t& opts, cv::Mat& vessel, cv::Mat& scale, cv::Mat& angle){
    std::vector<cv::Mat> ALLfiltered;
    std::vector<cv::Mat> ALLangles;

    float beta =    2 * opts.BetaOne * opts.BetaOne;
    float c =       2 * opts.BetaTwo * opts.BetaTwo;

    for (float sigma = opts.sigma_start; sigma <= opts.sigma_end; sigma += opts.sigma_step){
        //create 2D hessians
        cv::Mat Dxx, Dyy, Dxy;
        frangi2d_hessian(src, sigma, Dxx, Dxy, Dyy);

        //correct for scale
        Dxx = Dxx * sigma * sigma;
        Dyy = Dyy * sigma * sigma;
        Dxy = Dxy * sigma * sigma;

        //calculate (abs sorted) eigenvalues and vectors
        cv::Mat lambda1, lambda2, Ix, Iy;
        frangi2_eig2image(Dxx, Dxy, Dyy, lambda1, lambda2, Ix, Iy);

        //compute direction of the minor eigenvector
        cv::Mat angles;
        cv::phase(Ix, Iy, angles);
        ALLangles.push_back(angles);

        //compute some similarity measures
        lambda2.setTo(nextafterf(0, 1), lambda2 == 0);
        cv::Mat Rb = lambda1.mul(1.0/lambda2);
        Rb = Rb.mul(Rb);
        cv::Mat S2 = lambda1.mul(lambda1) + lambda2.mul(lambda2);

        //compute output image
        cv::Mat tmp1, tmp2;
        cv::exp(-Rb/beta, tmp1);
        cv::exp(-S2/c, tmp2);

        cv::Mat Ifiltered = tmp1.mul(cv::Mat::ones(src.rows, src.cols, src.type()) - tmp2);
        if (opts.BlackWhite){
            Ifiltered.setTo(0, lambda2 < 0);
        } else {
            Ifiltered.setTo(0, lambda2 > 0);
        }

        //store results
        ALLfiltered.push_back(Ifiltered);
    }

    float sigma = opts.sigma_start;
    ALLfiltered[0].copyTo(vessel);
    ALLfiltered[0].copyTo(scale);
    ALLfiltered[0].copyTo(angle);
    scale.setTo(sigma);

    //find element-wise maximum across all accumulated filter results
    for (int i=1; i < ALLfiltered.size(); i++){
        vessel = max(vessel, ALLfiltered[i]);
        scale.setTo(sigma, ALLfiltered[i] == vessel);
        ALLangles[i].copyTo(angle, ALLfiltered[i] == vessel);
        sigma += opts.sigma_step;
    }
}


/// Runs a 2d hessian filter on the source image using given sigma
void frangi2d_hessian(const cv::Mat& src, float sigma, cv::Mat& Dxx, cv::Mat& Dxy, cv::Mat& Dyy){
	// 1) Construct Hessian kernels
	int n_kern_x = 2 * round(3 * sigma) + 1;
	int n_kern_y = n_kern_x;

    // Float array initialization
    float* kern_xx_f = new float[n_kern_x * n_kern_y]();
    float* kern_xy_f = new float[n_kern_x * n_kern_y]();
    float* kern_yy_f = new float[n_kern_x * n_kern_y]();

    // TODO: parallelize loop!
	int i = 0;
	int j = 0;
	for (int x = -round(3*sigma); x <= round(3*sigma); x++, j = 0){
		for (int y = -round(3*sigma); y <= round(3*sigma); y++){
			kern_xx_f[i * n_kern_y + j] = 1.0f / (2.0f * M_PI * sigma * sigma * sigma * sigma) * (x*x / (sigma * sigma) - 1) * exp(-(x*x + y*y)/(2.0f*sigma*sigma));
			kern_xy_f[i * n_kern_y + j] = 1.0f / (2.0f * M_PI * sigma * sigma * sigma * sigma * sigma * sigma) * (x*y) * exp(-(x*x + y*y)/(2.0f*sigma*sigma));
			j++;
		}
		i++;
	}

    // TODO: parallelize loop!
	for (int j = 0; j < n_kern_y; j++){
		for (int i = 0; i < n_kern_x; i++){
			kern_yy_f[j * n_kern_x + i] = kern_xx_f[i * n_kern_x + j];
		}
	}


	// 2) Flip kernels since kernels aren't symmetric and opencv's filter2D operation performs a correlation, not a convolution
    cv::Mat kern_xx, kern_xy, kern_yy;

#ifndef NO_OMP
    // Parallel kernel flipping
    #pragma omp parallel sections default(none), shared(kern_xx, kern_xx_f, kern_xy, kern_xy_f, kern_yy, kern_yy_f, n_kern_x, n_kern_y)
    {
        #pragma omp section
        {
            cv::flip(cv::Mat(n_kern_y, n_kern_x, CV_32FC1, kern_xx_f), kern_xx, -1);
	    }

        #pragma omp section
        {
            cv::flip(cv::Mat(n_kern_y, n_kern_x, CV_32FC1, kern_xy_f), kern_xy, -1);
        }

        #pragma omp section
        {
            cv::flip(cv::Mat(n_kern_y, n_kern_x, CV_32FC1, kern_yy_f), kern_yy, -1);
        }
	}
#else
    // Sequential kernel flipping
    cv::flip(cv::Mat(n_kern_y, n_kern_x, CV_32FC1, kern_xx_f), kern_xx, -1);
    cv::flip(cv::Mat(n_kern_y, n_kern_x, CV_32FC1, kern_xy_f), kern_xy, -1);
    cv::flip(cv::Mat(n_kern_y, n_kern_x, CV_32FC1, kern_yy_f), kern_yy, -1);
#endif


	// 3) Run image filter using anchor since we are to perform a convolution, not a correlation
    cv::Point anchor(n_kern_x - n_kern_x/2 - 1, n_kern_y - n_kern_y/2 - 1);

#ifndef NO_OMP
    // Parallel filtering on image
    #pragma omp parallel sections default(none) shared(src, Dxx, Dxy, Dyy, kern_xx, kern_xy, kern_yy, anchor)
    {
        #pragma omp section
        {
            cv::filter2D(src, Dxx, -1, kern_xx, anchor);
        }

        #pragma omp section
        {
            cv::filter2D(src, Dxy, -1, kern_xy, anchor);
        }

        #pragma omp section
        {
            cv::filter2D(src, Dyy, -1, kern_yy, anchor);
        }
	}
#else
    // Sequential filtering on image
    cv::filter2D(src, Dxx, -1, kern_xx, anchor);
    cv::filter2D(src, Dxy, -1, kern_xy, anchor);
    cv::filter2D(src, Dyy, -1, kern_yy, anchor);
#endif


    // 4) Delete kernels
	delete[] kern_xx_f;
	delete[] kern_xy_f;
	delete[] kern_yy_f;
}


/// Sets given options to the default values
void frangi2d_createopts(frangi2d_opts_t& opts){
	opts.sigma_start    = DEFAULT_SIGMA_START;
	opts.sigma_end      = DEFAULT_SIGMA_END;
	opts.sigma_step     = DEFAULT_SIGMA_STEP;
	opts.BetaOne        = DEFAULT_BETA_ONE;     // ignore blob-like structures?
	opts.BetaTwo        = DEFAULT_BETA_TWO;     // appropriate background suppression for this specific image, but can change.
	opts.BlackWhite     = DEFAULT_BLACKWHITE;
}


/// Estimates eigenvalues from Dxx, Dxy, Dyy to the results lambda1, lambda2, Ix, Iy
void frangi2_eig2image(const cv::Mat& Dxx, const cv::Mat& Dxy, const cv::Mat& Dyy, cv::Mat& lambda1, cv::Mat& lambda2,
                        cv::Mat& Ix, cv::Mat& Iy){
	// 1) Calculate eigenvectors from image, v1 and v2
	cv::Mat v2x, v2y, tmp;
    {
        // Block scoped because tmp2 is never used afterwards
        cv::Mat tmp2 = Dxx - Dyy;
        cv::sqrt(tmp2.mul(tmp2) + 4*Dxy.mul(Dxy), tmp);

        // Sequential matrix addition
        v2x = 2*Dxy;
        v2y = Dyy - Dxx + tmp;
    }


	// 2) Normalize vectors
    {
        // Block scoped because mag is never used afterwards
        cv::Mat mag;
	    cv::sqrt((v2x.mul(v2x) + v2y.mul(v2y)), mag);

#ifndef NO_OMP
        // Parallel matrix multiplication/ copying
        #pragma omp parallel sections default(none) shared(v2x, v2y, mag)
        {
            #pragma omp section
            {
                cv::Mat v2xtmp = v2x.mul(1.0f/mag);
                v2xtmp.copyTo(v2x, mag != 0);
            }

            #pragma omp section
            {
                cv::Mat v2ytmp = v2y.mul(1.0f/mag);
                v2ytmp.copyTo(v2y, mag != 0);
            }
        }
#else
        // Sequential matrix multiplication/ copying
        cv::Mat v2tmp = v2x.mul(1.0f/mag);
        v2tmp.copyTo(v2x, mag != 0);

        v2tmp = v2y.mul(1.0f/mag);
        v2tmp.copyTo(v2y, mag != 0);
#endif
    }


	// 3) Eigenvectors are orthogonal
    cv::Mat v1x, v1y;
	v2y.copyTo(v1x);
	v1x = -1 * v1x;
	v2x.copyTo(v1y);

	// 4) Compute eigenvalues
    cv::Mat mu1 = 0.5*(Dxx + Dyy + tmp);
    cv::Mat mu2 = 0.5*(Dxx + Dyy - tmp);


	// 5) Sort eigenvalues by absolute value abs(Lambda1) < abs(Lamda2)
    const cv::Mat check = cv::abs(mu1) > cv::abs(mu2);

#ifndef NO_OMP
    // Parallel matrix copying
    #pragma omp parallel sections default(none) private(lambda1, lambda2, Ix, Iy) shared(mu1, mu2, v1x, v2x, v1y, v2y)
    {
        // lambda1 (output) section
        #pragma omp section
        {
            mu1.copyTo(lambda1);
            mu2.copyTo(lambda1, check);
        }

        // lambda2 (output) section
        #pragma omp section
        {
            mu2.copyTo(lambda2);
            mu1.copyTo(lambda2, check);
        }

        // Ix (output) section
        #pragma omp section
        {
            v1x.copyTo(Ix);
            v2x.copyTo(Ix, check);
        }

        // Iy (output) section
        #pragma omp section
        {
            v1y.copyTo(Iy);
            v2y.copyTo(Iy, check);
        }
    }
#else
    // Sequential matrix copying
    mu1.copyTo(lambda1);
    mu2.copyTo(lambda1, check);

    mu2.copyTo(lambda2);
    mu1.copyTo(lambda2, check);

    v1x.copyTo(Ix);
    v2x.copyTo(Ix, check);

    v1y.copyTo(Iy);
    v2y.copyTo(Iy, check);
#endif
}
