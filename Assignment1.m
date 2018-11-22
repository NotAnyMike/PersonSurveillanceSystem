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

load('data/person_re-identification/person_re-id_train.mat')

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

extract_settings = struct;

% parameters for correlogram
extract_settings.correlogram.use = false;
extract_settings.correlogram.window = [128 64];
extract_settings.correlogram.fun = @(block_struct) colorAutoCorrelogram(block_struct);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% parameters for colour histogram - RGB
extract_settings.col_hist_rgb.use = false;
extract_settings.col_hist_rgb.use_total = false;
extract_settings.col_hist_rgb.nbin = 30;
extract_settings.col_hist_rgb.window = [32 32];
extract_settings.col_hist_rgb.fun = @(block_struct) histcounts(block_struct.data, extract_settings.col_hist_rgb.nbin);

% parameters for colour histogram - HSV
extract_settings.col_hist_hsv.use = true;
extract_settings.col_hist_hsv.use_total = true;
extract_settings.col_hist_hsv.nbin = 80;
extract_settings.col_hist_hsv.window = [32 16];
extract_settings.col_hist_hsv.h_fun = @(block_struct) histcounts(block_struct.data, 60);
extract_settings.col_hist_hsv.s_fun = @(block_struct) histcounts(block_struct.data, 15);
extract_settings.col_hist_hsv.v_fun = @(block_struct) histcounts(block_struct.data, 30);
% good ratio is 8 : 2 : 4
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% parameters for HoG
extract_settings.hog.use = true;
extract_settings.hog.window = [16 16];
extract_settings.hog.nbins = 30;
extract_settings.hog.block_size = [4 4];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% parameters for LBP
extract_settings.lbp.use = true;
extract_settings.lbp.window = [16 16];
extract_settings.lbp.n_neighbour = 8;
extract_settings.lbp.radius = 1;
extract_settings.lbp.is_upright = true;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% parameters for SIFT
extract_settings.sift.use = false;
extract_settings.sift.step = 5;
extract_settings.sift.nbin = 8;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% parameters for MSCR
extract_settings.mscr.use = false;
extract_settings.mscr.mask = ones(128, 64);
extract_settings.mscr.mask = extract_settings.mscr.mask(:, :, 1);
extract_settings.mscr.p.min_margin=0.003; %  0.0015;  % Set margin parameter
extract_settings.mscr.p.ainc = 1.05;
extract_settings.mscr.p.min_size = 40;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 
%%%%% Feature extraction of X_train_1

nbins = [10 15 20 30 40 50];
windows = [16 16; 32 16; 32 32];

% for w = 1:length(windows)
%     extract_settings.col_hist_rgb.window = windows(w,:);
%     
%     for n = 1:length(nbins)
%         extract_settings.col_hist_rgb.nbin = nbins(n);
%         extract_settings.col_hist_rgb.fun = @(block_struct) histcounts(block_struct.data, extract_settings.col_hist_rgb.nbin);
        
X_train_1 = [];

for i = 1:length(image1)
    img = image1{i};
    X_train_1(i, :) = ExtractFeaturesPart1(img, extract_settings);
end

disp("Starting X_train_2")

% Feature extraction for X_train_2
X_train_2 = [];

for i = 1:length(image2)
    img = image2{i};
    X_train_2(i, :) = ExtractFeaturesPart1(img, extract_settings);
end


X_train = double(abs(X_train_1 - X_train_2));


% Train the re-identifier and evaluate the performance
%==========================================================================
% Try to train a better classifier.
%==========================================================================#
disp('Fitting model')
rng default
model = fitcsvm(X_train, Y_train, 'KernelScale','auto','Standardize',true,'OutlierFraction',0.09, 'Nu',0.25);

save('person_reid_model.mat','model');
save('person_reid_settings.mat', 'extract_settings');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loading the testing data
% - query, a 1 x N struct array with fields image and id
% - gallery, a 1 x N struct array with fiedls image and id
% - In the testing, we aim at re-identifying the query person in the
% gallery images.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('data/person_re-identification/person_re-id_test.mat')

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
    X_query(i, :) = ExtractFeaturesPart1(img, extract_settings);
end

% Feature extraction for X_query
X_gallery = [];

for i = 1:length(image_gallery)
    img = image_gallery{i};
    X_gallery(i, :) = ExtractFeaturesPart1(img, extract_settings);
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
% fprintf(' using settings window = %d %d, nbin = %d\n', windows(w, 1), windows(w,2), nbins(n));
fprintf('The mean Average Precision of person re-identification is:%.2f \n\n', mAP * 100)
       
%     end
% end


%% Visualization the result of person re-id

% query_idx = [2,15,26]; 
% gallery_idx = [2, 1, 45];
% l_ = reshape(l(:), [90, 30])';
% Y_test_ = reshape(Y_test, [90, 30])';
% nPairs = 3; % number of visualize data. maximum is 3
% % nPairs = length(data_idx); 
% visualise_reid(image_query, image_gallery, query_idx, gallery_idx, l_, Y_test_, nPairs )

