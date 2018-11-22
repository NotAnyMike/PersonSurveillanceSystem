% Image and Visual Computing Assignment 2: Person Attribute Recognition
%==========================================================================
%   In this assignment, you are expected to use the previous learned method
%   to cope with person attribute recognition problem. The vl_feat, 
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
clear all
clc
run ICV_setup

% Hyperparameter of experiments
resize_size=[128 64];

%% Part II: Person Attribute Recognition: 
%==========================================================================
% The aim of this task is to recognize the attribute of persons (e.g. gender) 
% in the images. We train several binary classifiers to predict 
% whether the person has a certain attribute (e.g. gender) or not.
% - Extract the features
% - Get a data representation for training
% - Train the recognizer and evaluate its performance
%==========================================================================


disp('Person Attribute:Extracting features..')
load('./data/person_attribute_recognition/person_attribute_tr.mat')

Xtr = [];
Xte = [];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loading the training data
% -tr_img/te_img:
% The data is store in a N-by-1 cell array. Each cell is a person image.
% -Ytr/Yte: is a struct where Ytr.bag is a N-by-1 vector of 'Yes' or 'No'
% In this assignment, there are six types of attributes to be recognized: 
% backpack, bag, gender, hat, shoes, upred
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% You need to use your own feature extractor by modifying the ExtractFeatureReid()
%  function or implement your own feature extraction function.
%  For example, use the BoW visual representation (Or any other better representation)


load('./data/person_attribute_recognition/person_attribute_te.mat')

% Old code:
% Functions with name ExtractFeatureAttribute[type].m are the original
% functions implemented to extract features using [type], but given that
% the best performance comes from simple features the following code
% incorpors all the functions and it is much more readable, altough is
% less efficient on unsued feature extractors.
% 
% The different ExtracFeature files are "hog" "bow_hog" "bow_sift" "sift"
% "color" and "lbp".


useCrossVal = 'off'; % "on" or "off"

% New Code:
% The following is the full configuration variable that contains all
% variables to run the model, by default it will only use histograms of rgb
% and hsv colors because they give the best performance 44% mAP, to turn on
% any other feature extraction use the flag 'use', the default parameters
% should be ok, but they are not the best and haven't been tune correctly,
% only the defaults paramenter for the two methods used (mentioned above)
% are tuned.

% Setting should look like:
dsettings = struct;
% color histogram
dsettings.color_hist.use        = true;
dsettings.color_hist.nbins      = 5;
dsettings.color_hist.win_size   = [16 16];
% hsv histogram
dsettings.col_hist_hsv.use      = true;
dsettings.col_hist_hsv.nbins    = 5;
dsettings.col_hist_hsv.win_size = [16 16];
% Sift Features
dsettings.sift.use     = false;
dsettings.sift.step_p  = 5;
dsettings.sift.binSize = 8;
%lbp
dsettings.lbp.use         = false;
dsettings.lbp.win_size    = [8 8];
dsettings.lbp.n_neighbour = 8;
dsettings.lbp.radius      = 1;
dsettings.lbp.is_upright  = true;
%BoW with Sift Features
dsettings.bow_with_sift.use          = false;
dsettings.bow_with_sift.step         = 3;
dsettings.bow_with_sift.binSize      = 3;
dsettings.bow_with_sift.num_of_words = 200;
dsettings.bow_with_sift.vocab        = 0;
%BoW with HoG features
dsettings.bow_with_hog.use          = false;
dsettings.bow_with_hog.num_of_words = 200;
dsettings.bow_with_hog.NumBins      = 18;
dsettings.bow_with_hog.BlockSize    = [2 2];
dsettings.bow_with_hog.CellSize     = [8 8];
dsettings.bow_with_hog.num_blocks   = 3;
dsettings.bow_with_hog.vocab        = 0;

% Loading vocabulary in case bow_with_sift will be used
if dsettings.bow_with_sift.use
    if ~exist('bow_sift_vocab.mat', 'file')
        fprintf('No existing visual word vocabulary found. Computing one from training images\n')
        vocab_sift = codebook_sift(tr_img, dsettings.bow_with_sift.num_of_words, dsettings.bow_with_sift.step, dsettings.bow_with_sift.binSize);
        save('bow_sift_vocab.mat', 'vocab_sift');
        load('bow_sift_vocab.mat');
    else 
        fprintf('Loading existing vocabulary\n')
        load('bow_sift_vocab.mat');
    end
    dsettings.bow_with_sift.vocab = vocab_sift;
end

% Loading another vocabulary in case bow_with_hog will be used
if dsettings.bow_with_hog.use
    if ~exist('bow_hog_vocab.mat', 'file')
        fprintf('No existing visual word vocabulary found. Computing one from training images\n')
        vocab_hog = codebook(tr_img, dsettings.bow_with_hog.num_of_words, dsettings.bow_with_hog.NumBins, dsettings.bow_with_hog.BlockSize, dsettings.bow_with_hog.CellSize, dsettings.bow_with_hog.num_blocks);
        save('bow_hog_vocab.mat', 'vocab_hog');
        load('bow_hog_vocab.mat');
    else 
        fprintf('Loading existing vocabulary\n')
        load('bow_hog_vocab.mat');
    end
    dsettings.bow_with_hog.vocab = vocab_hog;
end

Xtr = ExtractFeatureReid(tr_img, dsettings);
Xte = ExtractFeatureReid(te_img, dsettings);

Xtr = double(Xtr);
Xte = double(Xte);

if strcmp(useCrossVal, 'on')
    fprintf('Using crossvalidation\n')
end

% Train the recognizer and evaluate the performance
%% backpack
fprintf('Training backpack classifier...\n')


model.backpack = fitcsvm(Xtr,Ytr.backpack,'CrossVal',useCrossVal);


if strcmp(useCrossVal, 'off')
    [l.backpack,prob.backpack] = predict(model.backpack,Xte);
else
    predicted = [];
    probabilities1 = [];
    probabilities2 = [];
    for i=1:10
        curr_model = model.backpack.Trained{i};
        [class_pred,prob_pred] = predict(curr_model,Xte);
        predicted = [predicted, class_pred];
        probabilities1 = [probabilities1, prob_pred(:,1)];
        probabilities2 = [probabilities2, prob_pred(:,2)];
    end
    l.backpack = mode(predicted')';
    probabilities1_avg = mean(probabilities1')';
    probabilities2_avg = mean(probabilities2')';
    prob.backpack = [probabilities1_avg,probabilities2_avg];
end
    
% Compute the accuracy
acc.backpack = mean(l.backpack==Yte.backpack)*100;

fprintf('The accuracy of backpack recognition is:%.2f \n', acc.backpack)

%% bag
fprintf('Training bag classifier...\n')

model.bag = fitcsvm(Xtr,Ytr.bag,'CrossVal',useCrossVal);
if strcmp(useCrossVal, 'off')
    [l.bag,prob.bag] = predict(model.bag,Xte);
else
    predicted = [];
    probabilities1 = [];
    probabilities2 = [];
    for i=1:10
        curr_model = model.bag.Trained{i};
        [class_pred,prob_pred] = predict(curr_model,Xte);
        predicted = [predicted, class_pred];
        probabilities1 = [probabilities1, prob_pred(:,1)];
        probabilities2 = [probabilities2, prob_pred(:,2)];
    end
    l.bag = mode(predicted')';
    probabilities1_avg = mean(probabilities1')';
    probabilities2_avg = mean(probabilities2')';
    prob.bag = [probabilities1_avg,probabilities2_avg];
end

% Compute the accuracy
acc.bag = mean(l.bag==Yte.bag)*100;

fprintf('The accuracy of bag recognition is:%.2f \n', acc.bag)

%% gender
fprintf('Training gender classifier...\n')
model.gender = fitcsvm(Xtr,Ytr.gender,'CrossVal',useCrossVal);
if strcmp(useCrossVal, 'off')
    [l.gender,prob.gender] = predict(model.gender,Xte);
else
    predicted = [];
    probabilities1 = [];
    probabilities2 = [];
    for i=1:10
        curr_model = model.gender.Trained{i};
        [class_pred,prob_pred] = predict(curr_model,Xte);
        predicted = [predicted, class_pred];
        probabilities1 = [probabilities1, prob_pred(:,1)];
        probabilities2 = [probabilities2, prob_pred(:,2)];
    end
    l.gender = mode(predicted')';
    probabilities1_avg = mean(probabilities1')';
    probabilities2_avg = mean(probabilities2')';
    prob.gender = [probabilities1_avg,probabilities2_avg];
end

% Compute the accuracy
acc.gender = mean(l.gender==Yte.gender)*100;

fprintf('The accuracy of gender recognition is:%.2f \n', acc.gender)

%% hat
fprintf('Training hat classifier...\n')
model.hat = fitcsvm(Xtr,Ytr.hat,'CrossVal',useCrossVal);
if strcmp(useCrossVal, 'off')
    [l.hat,prob.hat] = predict(model.hat,Xte);
else
    predicted = [];
    probabilities1 = [];
    probabilities2 = [];
    for i=1:10
        curr_model = model.hat.Trained{i};
        [class_pred,prob_pred] = predict(curr_model,Xte);
        predicted = [predicted, class_pred];
        probabilities1 = [probabilities1, prob_pred(:,1)];
        probabilities2 = [probabilities2, prob_pred(:,2)];
    end
    l.hat = mode(predicted')';
    probabilities1_avg = mean(probabilities1')';
    probabilities2_avg = mean(probabilities2')';
    prob.hat = [probabilities1_avg,probabilities2_avg];
end

% Compute the accuracy
acc.hat = mean(l.hat==Yte.hat)*100;

fprintf('The accuracy of hat recognition is:%.2f \n', acc.hat)

%% shoes
fprintf('Training shoes classifier...\n')
model.shoes = fitcsvm(Xtr,Ytr.shoes,'CrossVal',useCrossVal);
if strcmp(useCrossVal, 'off')
    [l.shoes,prob.shoes] = predict(model.shoes,Xte);
else
    predicted = [];
    probabilities1 = [];
    probabilities2 = [];
    for i=1:10
        curr_model = model.shoes.Trained{i};
        [class_pred,prob_pred] = predict(curr_model,Xte);
        predicted = [predicted, class_pred];
        probabilities1 = [probabilities1, prob_pred(:,1)];
        probabilities2 = [probabilities2, prob_pred(:,2)];
    end
    l.shoes = mode(predicted')';
    probabilities1_avg = mean(probabilities1')';
    probabilities2_avg = mean(probabilities2')';
    prob.shoes = [probabilities1_avg,probabilities2_avg];
end

% Compute the accuracy
acc.shoes = mean(l.shoes==Yte.shoes)*100;

fprintf('The accuracy of shoes recognition is:%.2f \n', acc.shoes)

%% upred
fprintf('Training upred classifier...\n')
model.upred = fitcsvm(Xtr,Ytr.upred,'CrossVal',useCrossVal);
if strcmp(useCrossVal, 'off')
    [l.upred,prob.upred] = predict(model.upred,Xte);
else
    predicted = [];
    probabilities1 = [];
    probabilities2 = [];
    for i=1:10
        curr_model = model.upred.Trained{i};
        [class_pred,prob_pred] = predict(curr_model,Xte);
        predicted = [predicted, class_pred];
        probabilities1 = [probabilities1, prob_pred(:,1)];
        probabilities2 = [probabilities2, prob_pred(:,2)];
    end
    l.upred = mode(predicted')';
    probabilities1_avg = mean(probabilities1')';
    probabilities2_avg = mean(probabilities2')';
    prob.upred = [probabilities1_avg,probabilities2_avg];
end

% Compute the accuracy
acc.upred = mean(l.upred==Yte.upred)*100;

fprintf('The accuracy of upred recognition is:%.2f \n', acc.upred)

ave_acc = (acc.backpack + acc.bag + acc.gender + acc.hat + acc.shoes + acc.upred) / 6;

fprintf('The average accuracy of attribute recognition is:%.2f \n', ave_acc)


%% Compute the AP
AP = zeros(6,1);
%% backpack

% Compute the AP of searching the people with backpack
index = 1:length(Yte.backpack);
same_index = index(Yte.backpack==1);
[~, index] = sort(prob.backpack(:,2), 'descend');
[AP(1), ~] = compute_AP(same_index, index);

fprintf('The Average Precision of backpack retrieval is:%.2f \n', AP(1))

%% bag

% Compute the AP of searching the people with bag
index = 1:length(Yte.bag);
same_index = index(Yte.bag==1);
[~, index] = sort(prob.bag(:,2), 'descend');
[AP(2), ~] = compute_AP(same_index, index);

fprintf('The Average Precision of bag retrieval is:%.2f \n', AP(2))

%% gender

% Compute the AP of female people retrieval
index = 1:length(Yte.gender);
same_index = index(Yte.gender==1);
[~, index] = sort(prob.gender(:,2), 'descend');
[AP(3), ~] = compute_AP(same_index, index);

fprintf('The Average Precision of female retrieval is:%.2f \n', AP(3))

%% hat

% Compute the AP of hat retrieval
index = 1:length(Yte.hat);
same_index = index(Yte.hat==1);
[~, index] = sort(prob.hat(:,2), 'descend');
[AP(4), ~] = compute_AP(same_index, index);

fprintf('The Average Precision of hat retrieval is:%.2f \n', AP(4))

%% shoes

% Compute the AP of shoes retrieval
index = 1:length(Yte.shoes);
same_index = index(Yte.shoes==1);
[~, index] = sort(prob.shoes(:,2), 'descend');
[AP(5), ~] = compute_AP(same_index, index);

fprintf('The Average Precision of shoes retrieval is:%.2f \n', AP(5))

%% upred

% Compute the AP of upred retrieval
index = 1:length(Yte.upred);
same_index = index(Yte.upred==1);
[~, index] = sort(prob.upred(:,2), 'descend');
[AP(6), ~] = compute_AP(same_index, index);

fprintf('The Average Precision of upred people retrieval is:%.2f \n', AP(6))

mAP = mean(AP);

fprintf('The average accuracy of attribute recognition is:%.2f \n', mAP)



%% Visualization the result of person re-id

data_idx = [12,34,213]; % The index of image in validation set
nPairs = 3; % number of visualize data. maximum is 3
idx_attribute = 3;
nPairs = length(data_idx); 
visualise_attribute(te_img,prob,Yte,data_idx,nPairs, idx_attribute )
