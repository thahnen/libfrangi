#include <iostream>

#ifndef NO_OMP
#   include <omp.h>
#endif

#include "frangi.h"


/// Applys a full frangi filter to the source image using provided options
void frangi2d(const cv::Mat& src, cv::Mat& maxVals, cv::Mat& whatScale, cv::Mat& outAngles, frangi2d_opts_t opts){
    std::vector<cv::Mat> ALLfiltered;
    std::vector<cv::Mat> ALLangles;
    float beta = 2*opts.BetaOne*opts.BetaOne;
    float c = 2*opts.BetaTwo*opts.BetaTwo;

    for (float sigma = opts.sigma_start; sigma <= opts.sigma_end; sigma += opts.sigma_step){
        //create 2D hessians
        cv::Mat Dxx, Dyy, Dxy;
        frangi2d_hessian(src, sigma, Dxx, Dxy, Dyy);

        //correct for scale
        Dxx = Dxx*sigma*sigma;
        Dyy = Dyy*sigma*sigma;
        Dxy = Dxy*sigma*sigma;

        //calculate (abs sorted) eigenvalues and vectors
        cv::Mat lambda1, lambda2, Ix, Iy;
        frangi2_eig2image(Dxx, Dxy, Dyy, lambda1, lambda2, Ix, Iy);

        //compute direction of the minor eigenvector
        cv::Mat angles;
        phase(Ix, Iy, angles);
        ALLangles.push_back(angles);

        //compute some similarity measures
        lambda2.setTo(nextafterf(0, 1), lambda2 == 0);
        cv::Mat Rb = lambda1.mul(1.0/lambda2);
        Rb = Rb.mul(Rb);
        cv::Mat S2 = lambda1.mul(lambda1) + lambda2.mul(lambda2);

        //compute output image
        cv::Mat tmp1, tmp2;
        exp(-Rb/beta, tmp1);
        exp(-S2/c, tmp2);

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
    ALLfiltered[0].copyTo(maxVals);
    ALLfiltered[0].copyTo(whatScale);
    ALLfiltered[0].copyTo(outAngles);
    whatScale.setTo(sigma);

    //find element-wise maximum across all accumulated filter results
    for (int i=1; i < ALLfiltered.size(); i++){
        maxVals = max(maxVals, ALLfiltered[i]);
        whatScale.setTo(sigma, ALLfiltered[i] == maxVals);
        ALLangles[i].copyTo(outAngles, ALLfiltered[i] == maxVals);
        sigma += opts.sigma_step;
    }
}


/// Runs a 2d hessian filter on the source image using sigma
void frangi2d_hessian(const cv::Mat& src, float sigma, cv::Mat& Dxx, cv::Mat& Dxy, cv::Mat& Dyy){
	// 1) Construct Hessian kernels
	int n_kern_x = 2*round(3*sigma) + 1;
	int n_kern_y = n_kern_x;

	float* kern_xx_f, *kern_xy_f, *kern_yy_f;

#ifndef NO_OMP
	// Parallel float array initialization
    #pragma omp parallel sections default(none), private(kern_xx_f, kern_xy_f, kern_yy_f) shared(n_kern_x, n_kern_y)
    {
        #pragma omp section
        {
            kern_xx_f = new float[n_kern_x * n_kern_y]();
        }

        #pragma omp section
        {
            kern_xy_f = new float[n_kern_x * n_kern_y]();
        }

        #pragma omp section
        {
            kern_yy_f = new float[n_kern_x * n_kern_y]();
        }
    }
#else
    // Seqential float array initialization
    kern_xx_f = new float[n_kern_x * n_kern_y]();
    kern_xy_f = new float[n_kern_x * n_kern_y]();
    kern_yy_f = new float[n_kern_x * n_kern_y]();
#endif

	int i = 0;
	int j = 0;
	for (int x = -round(3*sigma); x <= round(3*sigma); x++){
		j=0;
		for (int y = -round(3*sigma); y <= round(3*sigma); y++){
			kern_xx_f[i*n_kern_y + j] = 1.0f/(2.0f*M_PI*sigma*sigma*sigma*sigma) * (x*x/(sigma*sigma) - 1) * exp(-(x*x + y*y)/(2.0f*sigma*sigma));
			kern_xy_f[i*n_kern_y + j] = 1.0f/(2.0f*M_PI*sigma*sigma*sigma*sigma*sigma*sigma)*(x*y)*exp(-(x*x + y*y)/(2.0f*sigma*sigma));
			j++;
		}
		i++;
	}

	for (int j=0; j < n_kern_y; j++){
		for (int i=0; i < n_kern_x; i++){
			kern_yy_f[j*n_kern_x + i] = kern_xx_f[i*n_kern_x + j];
		}
	}

	//flip kernels since kernels aren't symmetric and opencv's filter2D operation performs a correlation, not a convolution
    cv::Mat kern_xx;
	flip(cv::Mat(n_kern_y, n_kern_x, CV_32FC1, kern_xx_f), kern_xx, -1);

    cv::Mat kern_xy;
	flip(cv::Mat(n_kern_y, n_kern_x, CV_32FC1, kern_xy_f), kern_xy, -1);

    cv::Mat kern_yy;
	flip(cv::Mat(n_kern_y, n_kern_x, CV_32FC1, kern_yy_f), kern_yy, -1);

	//specify anchor since we are to perform a convolution, not a correlation
    cv::Point anchor(n_kern_x - n_kern_x/2 - 1, n_kern_y - n_kern_y/2 - 1);

	//run image filter
    cv::filter2D(src, Dxx, -1, kern_xx, anchor);
    cv::filter2D(src, Dxy, -1, kern_xy, anchor);
    cv::filter2D(src, Dyy, -1, kern_yy, anchor);


	delete[] kern_xx_f;
	delete[] kern_xy_f;
	delete[] kern_yy_f;
}


/// Sets given options to the default values
void frangi2d_createopts(frangi2d_opts_t& opts){
	//these parameters depend on the scale of the vessel, depending ultimately on the image size...
	opts.sigma_start =  DEFAULT_SIGMA_START;
	opts.sigma_end =    DEFAULT_SIGMA_END;
	opts.sigma_step =   DEFAULT_SIGMA_STEP;
	opts.BetaOne =      DEFAULT_BETA_ONE;       // ignore blob-like structures?
	opts.BetaTwo =      DEFAULT_BETA_TWO;       // appropriate background suppression for this specific image, but can change.
	opts.BlackWhite =   DEFAULT_BLACKWHITE;
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

#ifndef NO_OMP
        // Parallel matrix addition
        // TODO: Handle tmp (should not be shared as the first section does not need it)!
        #pragma omp parallel sections default(none) private(v2x, v2y) shared(Dxx, Dxy, Dyy, tmp)
        {
            #pragma omp section
            {
                v2x = 2*Dxy;
            }

            #pragma omp section
            {
                v2y = Dyy - Dxx + tmp;
            }
        }
#else
        // Sequential matrix addition
        v2x = 2*Dxy;
        v2y = Dyy - Dxx + tmp;
#endif
    }


	// 2) Normalize vectors
    {
        // Block scoped because mag is never used afterwards
        cv::Mat mag;
	    cv::sqrt((v2x.mul(v2x) + v2y.mul(v2y)), mag);

#ifndef NO_OMP
        // Parallel matrix multiplication/ copying
        #pragma omp parallel sections default(none) private(v2x, v2y) shared(mag)
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
	v1x = -1*v1x;
	v2x.copyTo(v1y);


	// 4) Compute eigenvalues
    cv::Mat mu1, mu2;

#ifndef NO_OMP
    // Parallel matrix initialization
    #pragma omp parallel sections default(none) private(mu1, mu2) shared(Dxx, Dxy, Dyy, tmp)
    {
        #pragma omp section
        {
            mu1 = 0.5*(Dxx + Dyy + tmp);
        }

        #pragma omp section
        {
            mu2 = 0.5*(Dxx + Dyy - tmp);
        }
    }
#else
    mu1 = 0.5*(Dxx + Dyy + tmp);
    mu2 = 0.5*(Dxx + Dyy - tmp);
#endif


	// 5) Sort eigenvalues by absolute value abs(Lambda1) < abs(Lamda2)
    cv::Mat check = abs(mu1) > abs(mu2);

#ifndef NO_OMP
    // Parallel matrix copying
    #pragma omp parallel sections default(none) private(mu1, mu2, lambda1, lambda2, v1x, v2x, Ix, v1y, v2y, Iy) shared(check)
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
