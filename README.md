# Person Surveillance System

Implementation of a Person Surveillance System on python, it should be able tell given two images of persons if they are the same person, the second module should be able to given some text attributes (hat, woman, etc.) retrive an images with such attributes

## Part 1 - Re-identification
The idea here is to take 2 pictures of people, extract features from both and see how they compare to each other. There are several possible ways to do that. Some discussed in the course are:
- Histogram of Gradients (HoG)
- Local binary patterns (LBP)
- Colour histogram

### Colour histogram Implementation
Cell wise colour histogram is extracted from image. It is parametrised based on number of bins in the histogram and the cell size. Let's play with both!
However, in Lecture 4.2.1, Tim suggests it's best to use colour histogram + HoG + LBP. Let's try that and maybe quote his paper?

Colour histogram parameters (window size 16):
bin=4 20.77%
bin=5 20.61%
bin=3 22.89%
bin=2 15.26%

Sticking with bin=3, let;s change the window size:
However, window size should be a able to devise the image size for best performance.
window=16 22.89%
window=8  11.88%
window=12 16.78%
window=20 21.22%
window=32  17.01%

So best parameters for colour histogram are window=16 and bin=3.
HSV colour histograms seem to do better than RGB ones. Why not use all?

HSV seems to favour 4 colour bins.

### Histogram of Gradients (HoG)
default params:
hog_win_size = [16 16];
hog_nbins = 24;
hog_block_size = [4 4];
Acc: 11.07%

Let's start playing with the bins - reduce them
hog_nbins = 24 11.07%
hog_nbins = 12 13.31%
hog_nbins = 6  15.50%
hog_nbins = 3  13.79%

Let's try to play with the block size now
hog_block_size = [4 4] 15.50%
hog_block_size = [3 3] 11.90%
hog_block_size = [3 3] 5.79%


### Local binary patterns (LBP)
default params
lbp_win_size = [16 16];
lbp_n_neighbour = 8;
Acc: 13.44%

Can't really change the window size, so let's play with the neighbours
lbp_n_neighbour=16 9.69%
lbp_n_neighbour=4 17.23%
lbp_n_neighbour=6 16.12%
lbp_n_neighbour=3 12.02%


### Tuning the SVM
With all of the 3 features from above concatenated
Outlier rejection with fraction 0.00 - 27.89%
Outlier rejection with fraction 0.05 - 27.19%
Outlier rejection with fraction 0.1 -  27.51%
Outlier rejection with fraction 0.15 - 25.67%
Outlier rejection with fraction 0.09 - 28.33%
nu = 0.1 - 27.64%
nu = 0.2 - 27.90%
nu = 0.3 - 27.75%
nu = 0.25 - 28.91%
nu = 0.22 - 28.98%

Tuned SVM will be refereed to as SVM+.

### Correlogram
Implementation used from the official MATLAB package. Increases performance on average by approx 2%.

### SIFT
Explored but produced bad results. Doesn't really make sesnse to use SIFT here as all our our images have similar orientation and scale. **Not used.**

### MSCR
Used by itself and produced OK results (~26%) but when combines with everything else it deteriorates. **Not used.**



### Putting it all together
Results:
| Feature extractor | Comparator | Avg. precision |
| -- | -- | -- |
| HoG | SVM | 11.07 |
| HoG + Colour histogram | SVM | 13.12 |
| HoG + Colour histogram + LBP | SVM | 13.14 |
| Colour histogram | SVM | 11.00 |
| Colour Histogram + HoG + LBP | SVM+ | 28.99 |
| Correlogram + Colour histogram + LBP | SVM-  | 31.07 |
| Correlogram + Colour histogram + HOG + LBP | SVM-one | 35.49 |


### Final parameters
```
% parameters for correlogram
use_correlogram = true;
correlogram_window = [128 64];
correlogram_fun = @(block_struct) colorAutoCorrelogram(block_struct);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% parameters for colour histogram
use_colour = false;
use_colour_hsv = true;
colour_nbin = 4;
colour_win_size = [16 16];
fun = @(block_struct) histcounts(block_struct.data,colour_nbin);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% parameters for HoG
use_hog = true;
hog_win_size = [16 16];
hog_nbins = 4;
hog_block_size = [2 2];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   %

% parameters for LGP
use_lgp = true;
lbp_win_size = [16 16];
lbp_n_neighbour = 8;
lgp_radius = 1;
is_upright = true;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
```


## Things to try

** Colors are very useful **

## Results (Module 2)

these are the results

| Configuration | Using CrossVal | mAP |
|:-------------:|:--------------:|:---:|
| Base System   | No | 33% |
| Hog Attributes with 16 wind and 24 bins and block of 4 | No | 37% |
| HoG Attributes | Yes | 35% |
| BoW with HOG attributes | No | 29% |
| BoW with SIFT attributes | No | 32% |
| BoW with SIFT attributes | Yes| 32% |
| SIFT attributes | Yes | 27% |
| SIFT attributes | No | 36% |
| Colors hist with norm | No | 32% |
| Window 16 of hist colors | No | 38% |
| Window 16 of hist colors with sum | No | 39% |
| Window 16 of hist colors with sum and total | No | 40% |
| LBP | | 36% |
