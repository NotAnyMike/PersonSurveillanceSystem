% Image and Visual Computing Assignment 1: Person Re-identification
%==========================================================================
%   In this assignment, you are expected to use the previous learned method
%   to cope with person re-identification problem. The vl_feat, 
%   libsvm, liblinear and any other classification and feature extraction 
%   library are allowed to use in this assignment. The built-in matlab 
%   object-detection functionis not allowed. Good luck and have fun!
%
%                                               Released Date:   7/11/2018
%==========================================================================

%% Initialisation
%==========================================================================
% Add the path of used library.
% - The function of adding path of liblinear and vlfeat is included.
%==========================================================================
clearvars
run ICV_setup

% Hyperparameter of experiments
resize_size=[128 64];


%% Part 1: Person Re-identification (re-id): 
%==========================================================================
% The aim of this task is to verify whether the two given people in the
% images are the same person. We train a binary classifier to predict
% whether these two people are actually the same person or not.
% - Extract the features
% - Get a data representation for training
% - Train the re-identifier and evaluate its performance
%==========================================================================


disp('Person Re-id:Extracting features..')

load('~/ivc/IVC_assignment-2018/data/person_re-identification/person_re-id_train.mat')

dir = pwd();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loading the training data
% - train, a 1 x N struct array with fields image1, id1, image2 and id2
% - image1 and image2 are two image while id1 and id2 are the corresponding
% label. Here, we generate a label vector (i.e. Ytr) which indicates image1 
% is the same person as the image2 or not, where, Ytr = 1 indicates 'same' 
% and Ytr = -1 represents 'different'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% You should construct the features in here. (read, resize, extract)
image1 = {train(:).image1}';
image2 = {train(:).image2}';
id1 = [train(:).id1]';
id2 = [train(:).id2]';
Y_train = ones(length(id1),1);
Y_train(id1 ~= id2) = -1;

%% You need to use your own feature extractor by modifying the ExtractFeatureReid()
%  function or implement your own feature extraction function.
%  For example, use the BoW visual representation (Or any other better representation)

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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% parameters for LGP
use_lgp = true;
lbp_win_size = [16 16];
lbp_n_neighbour = 8;
lgp_radius = 1;
is_upright = true;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% parameters for SIFT
use_sift = false;
sift_step = 5;
sift_nbin = 8;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% parameters for PCA
use_pca = false;
pca_dim = 200;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% parameters for MSCR
use_mscr = false;
B = ones(128, 64);
B = B(:, :, 1);
p.min_margin=0.003; %  0.0015;  % Set margin parameter
p.ainc = 1.05;
p.min_size = 40;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%% Feature extraction of X_train_1
X_train_1 = [];

for i = 1:length(image1)
    img = image1{i};
    img_grey = rgb2gray(img);
    tmp = [];
    
    if (use_correlogram)
        correlogram_features = blockproc(img, correlogram_window, correlogram_fun);
        tmp = [tmp, correlogram_features(:)'];
    end
    
    if (use_colour)
        r = blockproc(img(:,:,1), colour_win_size, fun);
        r = r(:)';
        g = blockproc(img(:,:,2), colour_win_size, fun);
        g = g(:)';
        b = blockproc(img(:,:,3), colour_win_size, fun);
        b = b(:)';
        colour_features = [r g b];
        colour_features = colour_features / sum(colour_features);
        tmp = [tmp, colour_features];
    end
    
    if (use_colour_hsv)
        img_hsv = rgb2hsv(img);
        h = blockproc(img_hsv(:,:,1), colour_win_size, fun);
        h = h(:)';
        s = blockproc(img_hsv(:,:,2), colour_win_size, fun);
        s = s(:)';
        v = blockproc(img_hsv(:,:,3), colour_win_size, fun);
        v = v(:)';
        t = histcounts(img_hsv, 5);
        colour_features = [h s t];
        colour_features = colour_features / sum(colour_features);
        tmp = [tmp, colour_features];
    end
    
    if (use_hog)
        hog_features = extractHOGFeatures(img_grey,'CellSize',hog_win_size,...
            'NumBins', hog_nbins, 'BlockSize', hog_block_size);
        tmp = [tmp, hog_features];
    end
    
    if (use_lgp)
        lgp_features = extractLBPFeatures(img_grey,'CellSize',lbp_win_size,'NumNeighbors',lbp_n_neighbour,...
            'Radius', lgp_radius, 'Upright', is_upright);
        tmp = [tmp, lgp_features];
    end
    
    if (use_sift)
        img_grey = imresize(img_grey, resize_size, 'bilinear');
        [sift_frames, sift_features] = vl_dsift(single(img_grey),'Step',sift_step,'size', sift_nbin,'fast');
        tmp = [tmp, sift_features(:)];
    end
    
    if (use_mscr)
        [mvec,pvec] = detect_mscr_masked(im2double(img),B,p);
        tmp = [tmp, mvec(:)', pvec(:)'];
    end       
    
    X_train_1(i, :) = tmp;
end

if (use_pca)
    coeff = pca(X_train_1);
    pca_dim_reduce = coeff(:, 1:pca_dim);
    X_train_1 = X_train_1 * pca_dim_reduce;
end

disp("Starting X_train_2")

% Feature extraction for X_train_2
X_train_2 = [];

for i = 1:length(image2)
    img = image2{i};
    img_grey = rgb2gray(img);
    tmp = [];
    
    if (use_correlogram)
        correlogram_features = blockproc(img, correlogram_window, correlogram_fun);
        tmp = [tmp, correlogram_features(:)'];
    end
    
    if (use_colour)
        r = blockproc(img(:,:,1), colour_win_size, fun);
        r = r(:)';
        g = blockproc(img(:,:,2), colour_win_size, fun);
        g = g(:)';
        b = blockproc(img(:,:,3), colour_win_size, fun);
        b = b(:)';
        colour_features = [r g b];
        colour_features = colour_features / sum(colour_features);
        tmp = [tmp, colour_features];
    end
    
    if (use_colour_hsv)
        img_hsv = rgb2hsv(img);
        h = blockproc(img_hsv(:,:,1), colour_win_size, fun);
        h = h(:)';
        s = blockproc(img_hsv(:,:,2), colour_win_size, fun);
        s = s(:)';
        v = blockproc(img_hsv(:,:,3), colour_win_size, fun);
        v = v(:)';
        t = histcounts(img_hsv, 5);
        colour_features = [h s t];
        colour_features = colour_features / sum(colour_features);
        tmp = [tmp, colour_features];
    end
    
    if (use_hog)
        hog_features = extractHOGFeatures(img_grey,'CellSize',hog_win_size, 'NumBins', hog_nbins, 'BlockSize', hog_block_size);
        tmp = [tmp, hog_features];
    end
    
    if (use_lgp)
        lgp_features = extractLBPFeatures(img_grey,'CellSize',lbp_win_size,'NumNeighbors',lbp_n_neighbour,...
            'Radius', lgp_radius, 'Upright', is_upright);
        tmp = [tmp, lgp_features];
    end
    
    if (use_sift)
        img_grey = imresize(img_grey, resize_size, 'bilinear');
        [sift_frames, sift_features] = vl_dsift(single(img_grey),'Step',sift_step,'size', sift_nbin,'fast');
        tmp = [tmp, sift_features(:)];
    end
    
    X_train_2(i, :) = tmp;
end

if (use_pca)
    X_train_2 = X_train_2 * pca_dim_reduce;
end

X_train = double(abs(X_train_1 - X_train_2));


% Train the re-identifier and evaluate the performance
%==========================================================================
% Try to train a better classifier.
%==========================================================================#
disp('Fitting model')

% rng default
% model = fitcsvm(X_train, Y_train,'OptimizeHyperparameters','auto',...
%      'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
%      'expected-improvement-plus'));

model = fitcsvm(X_train, Y_train, 'KernelScale','auto','Standardize',true,'OutlierFraction',0.09, 'Nu',0.22);
%       'OptimizeHyperparameters','auto',...
%       'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
%       'expected-improvement-plus'));

%model = fitcknn(X_train, Y_train,'NumNeighbors',3);
save('person_reid_model.mat','model');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loading the testing data
% - query, a 1 x N struct array with fields image and id
% - gallery, a 1 x N struct array with fiedls image and id
% - In the testing, we aim at re-identifying the query person in the
% gallery images.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('~/ivc/IVC_assignment-2018/data/person_re-identification/person_re-id_test.mat')

image_query = {query(:).image}';
id_query = [query(:).id]';
image_gallery = {gallery(:).image}';
id_gallery = [gallery(:).id]';
num_query = length(image_query);
num_gallery = length(image_gallery);

% Feature extraction for X_query
X_query = [];

for i = 1:length(image_query)
    img = image_query{i};
    img_grey = rgb2gray(img);
    tmp = [];
    
    if (use_correlogram)
        correlogram_features = blockproc(img, correlogram_window, correlogram_fun);
        tmp = [tmp, correlogram_features(:)'];
    end
    
    if (use_colour)
        r = blockproc(img(:,:,1), colour_win_size, fun);
        r = r(:)';
        g = blockproc(img(:,:,2), colour_win_size, fun);
        g = g(:)';
        b = blockproc(img(:,:,3), colour_win_size, fun);
        b = b(:)';
        colour_features = [r g b];
        colour_features = colour_features / sum(colour_features);
        tmp = [tmp, colour_features];
    end
    
    if (use_colour_hsv)
        img_hsv = rgb2hsv(img);
        h = blockproc(img_hsv(:,:,1), colour_win_size, fun);
        h = h(:)';
        s = blockproc(img_hsv(:,:,2), colour_win_size, fun);
        s = s(:)';
        v = blockproc(img_hsv(:,:,3), colour_win_size, fun);
        v = v(:)';
        t = histcounts(img_hsv, 5);
        colour_features = [h s, t];
        colour_features = colour_features / sum(colour_features);
        tmp = [tmp, colour_features];
    end
    
    if (use_hog)
        hog_features = extractHOGFeatures(img_grey,'CellSize',hog_win_size, 'NumBins', hog_nbins, 'BlockSize', hog_block_size);
        tmp = [tmp, hog_features];
    end
    
    if (use_lgp)
        lgp_features = extractLBPFeatures(img_grey,'CellSize',lbp_win_size,'NumNeighbors',lbp_n_neighbour,...
            'Radius', lgp_radius, 'Upright', is_upright);
        tmp = [tmp, lgp_features];
    end
    
    if (use_sift)
        img_grey = imresize(img_grey, resize_size, 'bilinear');
        [sift_frames, sift_features] = vl_dsift(single(img_grey),'Step',sift_step,'size', sift_nbin,'fast');
        tmp = [tmp, sift_features(:)];
    end
    
    X_query(i, :) = tmp;
end

if (use_pca)
    X_query = X_query * pca_dim_reduce;
end


% Feature extraction for X_query
X_gallery = [];

for i = 1:length(image_gallery)
    img = image_gallery{i};
    img_grey = rgb2gray(img);
    tmp = [];
    
    if (use_correlogram)
        correlogram_features = blockproc(img, correlogram_window, correlogram_fun);
        tmp = [tmp, correlogram_features(:)'];
    end
    
    if (use_colour)
        r = blockproc(img(:,:,1), colour_win_size, fun);
        r = r(:)';
        g = blockproc(img(:,:,2), colour_win_size, fun);
        g = g(:)';
        b = blockproc(img(:,:,3), colour_win_size, fun);
        b = b(:)';
        colour_features = [r g b];
        colour_features = colour_features / sum(colour_features);
        tmp = [tmp, colour_features];
    end
    
    if (use_colour_hsv)
        img_hsv = rgb2hsv(img);
        h = blockproc(img_hsv(:,:,1), colour_win_size, fun);
        h = h(:)';
        s = blockproc(img_hsv(:,:,2), colour_win_size, fun);
        s = s(:)';
        v = blockproc(img_hsv(:,:,3), colour_win_size, fun);
        v = v(:)';
        t = histcounts(img_hsv, 5);
        colour_features = [h s t];
        colour_features = colour_features / sum(colour_features);
        tmp = [tmp, colour_features];
    end
    
    if (use_hog)
        hog_features = extractHOGFeatures(img_grey,'CellSize',hog_win_size, 'NumBins', hog_nbins, 'BlockSize', hog_block_size);
        tmp = [tmp, hog_features];
    end
    
    if (use_lgp)
        lgp_features = extractLBPFeatures(img_grey,'CellSize',lbp_win_size,'NumNeighbors',lbp_n_neighbour,...
            'Radius', lgp_radius, 'Upright', is_upright);
        tmp = [tmp, lgp_features];
    end
    
    if (use_sift)
        img_grey = imresize(img_grey, resize_size, 'bilinear');
        [sift_frames, sift_features] = vl_dsift(single(img_grey),'Step',sift_step,'size', sift_nbin,'fast');
        tmp = [tmp, sift_features(:)];
    end
    
    X_gallery(i, :) = tmp;
end

if (use_pca)
    X_gallery = X_gallery * pca_dim_reduce;
end

% Constructing query-gallery pairs and label vector
X_test = [];
Y_test = [];
for i = 1:length(image_query)
    X_query_ = X_query(i,:);
    temp_X_test = abs(X_query_ - X_gallery);
    X_test = [X_test; temp_X_test];
    temp_Y_test = ones(num_gallery,1);
    temp_Y_test(id_gallery ~= id_query(i)) = -1;
    Y_test = [Y_test; temp_Y_test];
end

X_test = double(X_test);

%% 
disp("Predicting...")
[l,prob] = predict(model,X_test);
% [l, acc, prob] = svmpredict(Y_test, X_test, model);

% Compute the mean Average Precision
ap = zeros(num_query, 1);
for i = 1:num_query
    prob_i = prob((i - 1) * num_gallery + 1: i * num_gallery,2);
    [~, sorted_index] = sort(prob_i, 'descend');
    temp_index = 1:num_gallery;
    same_index = temp_index(id_gallery == id_query(i));
    [ap(i), ~] = compute_AP(same_index, sorted_index);
end
mAP = mean(ap);
fprintf('The mean Average Precision of person re-identification is:%.2f \n', mAP * 100)

%% Visualization the result of person re-id

query_idx = [2,15,26]; 
gallery_idx = [2, 1, 45];
l_ = reshape(l(:), [90, 30])';
Y_test_ = reshape(Y_test, [90, 30])';
nPairs = 3; % number of visualize data. maximum is 3
% nPairs = length(data_idx); 
visualise_reid(image_query, image_gallery, query_idx, gallery_idx, l_, Y_test_, nPairs )