% Image and Visual Computing Assignment 3: People Search
%==========================================================================
%   In this assignment, you are expected to use the previous learned method
%   to cope with people search. The vl_feat, libsvm, liblinear and
%   any other classification and feature extraction library are allowed to 
%   use in this assignment. The built-in matlab object-detection function
%   is not allowed. Good luck and have fun!
%
%                                               Released Date:   07/11/2018
%==========================================================================

%% Initialisation
%==========================================================================
% Add the path of used library.
% - The function of adding path of liblinear and vlfeat is included.
% - In the pedestrian_detection.mat, persons contains 6666 person images 
% while neg_persons are 6666 negative images. You can choose to use a small
% subset of them to train your own pedestrian detector.
% - In the person_search.mat, query is a struct array with fields: image,
% and id while gallery is a struct array with fields: image(whole image),
% frame_name, gt_bbox and id_bbox.
%==========================================================================
% clear all
clc
run ICV_setup
addpath('./detection_utils/')

% load the pedestrian detector training data
load('./data/people_search/pedestrian_detection/pedestrian_detection.mat');
% load the person search data
load('./data/people_search/people_search.mat');

% Hyperparameter of experiments
gallery_image_size = [540, 960];
resize_size_reid = [128, 64];
resize_detector = [150, 50];
threshold_detect = 0.2;    
threshold_nms = 0.5;
% num of positive training images
num_persons = 600;

%% pedestrian detection:
%==========================================================================
% Since you have your own pedestrian detector in Lab 4, you are expected
% to implement your own pedestrian detector by using the data in the
% pedestrian_detection.mat to train your detection model. After you train
% your pedestrian detector, you then can detect pedestrain in all gallery
% images by your detector. 
%                                (You should finish this part by yourself)
%==========================================================================

% extracting features
% You need to modify the ExtractFeature() function in '.\pedestrian detection\'
% to implement your own feature extractor. Note that if you have change the
% feature extraction function, remember to change the feature extractor in
% the window() function as well.
[pos_train,pos_label] = ExtractFeature(persons(1:num_persons), resize_detector,'pos');
[neg_train,neg_label] = ExtractFeature(neg_persons(1:num_persons), resize_detector, 'neg');

train_data = [pos_train;neg_train];
train_label = [pos_label;neg_label];

% training the detection model
svm_model1 = fitcsvm(train_data,train_label,'KernelFunction','gaussian');

clear persons neg_persons pos_train neg_train

image_gallery = {};
gallery_bbox = {};
gallery_score = {};

% initalise YOLO
letsyolo();
yolo_thresh = 0.1;
yolo_hier_thresh = 0.5;

num_gallery = length(gallery);

%% YOLO detect people

[gallery_score, gallery_bbox, ppl_images, detections] = yolo_detect(gallery,...
    gallery_image_size, yolo_thresh, yolo_hier_thresh);


%% Evaluating your result on the person search data via mAP
ap = zeros(num_gallery,1);
for i = 1:num_gallery
    ap(i) = evaluate(gallery(i).gt_bbox, gallery_bbox{i}, gallery_score{i});
end
mAP_detection = mean(ap);

fprintf('The mean Average Precision of pedestrian detection is:%.2f \n', mAP_detection * 100);

%% Seaching the query persons in detecting candidates/bounding boxes
%==========================================================================
% Since you have trained a person re-identification and you have detect
% pedestrian in all gallery images. Then the people search is to tell
% whether the detected pedestrian is the query person or not.
%                                (You should finish this part by yourself)
%==========================================================================

% load pretrained model with settings for feature extraction
load('person_reid_model.mat');
load('person_reid_settings.mat');

%% crop candidates from gallery images
id_gallery = [];
frame = [];
image_gallery = ppl_images;
num_gallery = length(gallery);
for i = 1:num_gallery
    id_ = get_id(gallery_bbox{i}, gallery(i).gt_bbox, gallery(i).id_bbox);
    id_gallery = [id_gallery; id_];
end

image_query = {query(:).image}';
id_query = [query(:).id]';
num_query = length(image_query);
num_gallery = length(image_gallery);

% Extracting query features
Xq = [];
for i = 1:length(image_query)
    Xq(i, :) = ExtractFeaturesPart1(image_query{i}, resize_size_reid, extract_settings);
end

% Extracting gallery features
Xg = [];
for i = 1:length(image_gallery)
    Xg(i, :) = ExtractFeaturesPart1(image_gallery{i}, resize_size_reid, extract_settings);
end

% Constructing query-gallery pairs and label vector
Xte = [];
Yte = [];
for i = 1:length(image_query)
    Xq_ = Xq(i,:);
    temp_Xte = Xq_ - Xg;
    Xte = [Xte; temp_Xte];
    temp_Yte = ones(num_gallery,1);
    temp_Yte(id_gallery ~= id_query(i)) = -1;
    Yte = [Yte; temp_Yte];
end

Xte = double(Xte);

[l,prob] = predict(model,Xte);

ap = zeros(num_query, 1);
for i = 1:num_query
    prob_i = prob((i - 1) * num_gallery + 1: i * num_gallery,2);
    [~, sorted_index] = sort(prob_i, 'descend');
    temp_index = 1:num_gallery;
    same_index = temp_index(id_gallery == id_query(i));
    if length(same_index) > 0
        [ap(i), ~] = compute_AP(same_index, sorted_index);
    else
        ap(i) = 0;
    end
end
mAP_reid = mean(ap);
fprintf('The mean Average Precision of person re-identification is:%.2f \n', mAP_reid * 100)


%% Try to visualize the results by yourself
%==========================================================================

for i = 1:length(image_gallery)
    draw_box(gallery(i).image, gallery(i).gt_bbox, gallery_bbox{i})
    pause(1)
end
