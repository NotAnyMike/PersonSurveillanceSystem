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

% objective is mAP 33%

% BoW visual representation (Or any other better representation)

model_name = "color"; % "hog" "bow" "bow_sift" "sift" "color" "lbp"
useCrossVal = 'off'; % "on" or "off"

if strcmp(model_name, 'bow')
    fprintf("using bow\n")
    num_of_words = 200;
    NumBins = 18;
    BlockSize = [2 2];
    CellSize = [8 8];
    num_blocks = 3;
    
    vocabulary = codebook(tr_img, num_of_words, NumBins, BlockSize, CellSize, num_blocks);
    
    Xtr = ExtractFeatureReid(tr_img, resize_size, vocabulary, NumBins, BlockSize, CellSize, num_blocks);
    Xte = ExtractFeatureReid(te_img, resize_size, vocabulary,NumBins, BlockSize, CellSize, num_blocks);
elseif strcmp(model_name, 'sift')
    fprintf("using sift\n")
    Xtr = ExtractFeatureAttributeSift(tr_img, resize_size);
    Xte = ExtractFeatureAttributeSift(te_img, resize_size);
elseif strcmp(model_name, 'bow_sift')
    fprintf("using bow sift\n")
    step = 3;
    binSize = 3;
    
    if ~exist('bow_sift_vocab.mat', 'file')
        fprintf('No existing visual word vocabulary found. Computing one from training images\n')

        num_of_words = 200;

        vocabulary = codebook_sift(tr_img, num_of_words, step, binSize);
        
        save('bow_sift_vocab.mat', 'vocabulary');
        load('bow_sift_vocab.mat');
    else 
        fprintf('Loading existing vocabulary\n')
        load('bow_sift_vocab.mat');
    end
    
    fprintf('Extracting features from sets\n')
    Xtr = ExtractFeatureAttributeBoWSift(tr_img, vocabulary, step, binSize);
    Xte = ExtractFeatureAttributeBoWSift(te_img, vocabulary, step, binSize);
elseif strcmp(model_name, 'lbp')
    fprintf('Using LBP\n')
    Xtr = ExtractFeatureAttributeLBP(tr_img);
    Xte = ExtractFeatureAttributeLBP(te_img);
elseif strcmp(model_name, "color")
    fprintf('Using Color\n')
    Xtr = ExtractFeatureAttributeColor(tr_img);
    Xte = ExtractFeatureAttributeColor(te_img);
elseif strcmp(model_name, "two")
    fprintf('Using two\n')
    Xtr_1 = ExtractFeatureAttributeColor(tr_img);
    Xte_1 = ExtractFeatureAttributeColor(te_img);
    
    Xtr_2 = ExtractFeatureAttribute(tr_img,resize_size);
    Xte_2 = ExtractFeatureAttribute(te_img,resize_size);
else
    fprintf("using default\n")
    [Xtr, ~] = ExtractFeatureAttribute(tr_img, resize_size);
    [Xte, ~] = ExtractFeatureAttribute(te_img, resize_size);
end

Xtr = double(Xtr);
Xte = double(Xte);

if strcmp(model_name, 'sift')
    Xtr = Xtr';
    Xte = Xte';
end

if strcmp(useCrossVal, 'on')
    fprintf('Using crossvalidation\n')
end

% Train the recognizer and evaluate the performance
%% backpack
fprintf('Training backpack classifier...\n')

if strcmp(model_name, "two")
    model.backpack_1 = fitcsvm(Xtr_1,Ytr.backpack);
    model.backpack_2 = fitcsvm(Xtr_2,Ytr.backpack);
else
    model.backpack = fitcsvm(Xtr,Ytr.backpack,'CrossVal',useCrossVal);
end

if strcmp(useCrossVal, 'off')
    if strcmp(model_name, "two")
        [l.backpack1,prob.backpack1] = predict(model.backpack_1,Xte_1);
        [l.backpack2,prob.backpack2] = predict(model.backpack_2,Xte_2);
        l.backpack = floor((l.backpack1+l.backpack2)/2);
        prob.backpack = max(prob.backpack1(:,2),prob.backpack2(:,2));
        prob.backpack = [-prob.backpack prob.backpack];
    else
        [l.backpack,prob.backpack] = predict(model.backpack,Xte);
    end
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
if strcmp(model_name, "two")
    model.bag_1 = fitcsvm(Xtr_1,Ytr.bag);
    model.bag_2 = fitcsvm(Xtr_2,Ytr.bag);
else
    model.bag = fitcsvm(Xtr,Ytr.bag,'CrossVal',useCrossVal);
end

if strcmp(useCrossVal, 'off')
    if strcmp(model_name, "two")
        [l.bag1,prob.bag1] = predict(model.bag_1,Xte_1);
        [l.bag2,prob.bag2] = predict(model.bag_2,Xte_2);
        l.bag = floor((l.bag1+l.bag2)/2);
        prob.bag = max(prob.bag1(:,2),prob.bag2(:,2));
        prob.bag = [-prob.bag prob.bag];
    else
        [l.bag,prob.bag] = predict(model.bag,Xte);
    end
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
if strcmp(model_name, "two")
    model.gender_1 = fitcsvm(Xtr_1,Ytr.gender);
    model.gender_2 = fitcsvm(Xtr_2,Ytr.gender);
else
    model.gender = fitcsvm(Xtr,Ytr.gender,'CrossVal',useCrossVal);
end

if strcmp(useCrossVal, 'off')
    if strcmp(model_name, "two")
        [l.gender1,prob.gender1] = predict(model.gender_1,Xte_1);
        [l.gender2,prob.gender2] = predict(model.gender_2,Xte_2);
        l.gender = floor((l.gender1+l.gender2)/2);
        prob.gender = max(prob.gender1(:,2),prob.gender2(:,2));
        prob.gender = [-prob.gender prob.gender];
    else
        [l.gender,prob.gender] = predict(model.gender,Xte);
    end
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
if strcmp(model_name, "two")
    model.hat_1 = fitcsvm(Xtr_1,Ytr.hat);
    model.hat_2 = fitcsvm(Xtr_2,Ytr.hat);
else
    model.hat = fitcsvm(Xtr,Ytr.hat,'CrossVal',useCrossVal);
end

if strcmp(useCrossVal, 'off')
    if strcmp(model_name, "two")
        [l.hat1,prob.hat1] = predict(model.hat_1,Xte_1);
        [l.hat2,prob.hat2] = predict(model.hat_2,Xte_2);
        l.hat = floor((l.hat1+l.hat2)/2);
        prob.hat = max(prob.hat1(:,2),prob.hat2(:,2));
        prob.hat = [-prob.hat prob.hat];
    else
        [l.hat,prob.hat] = predict(model.hat,Xte);
    end
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
if strcmp(model_name, "two")
    model.shoes_1 = fitcsvm(Xtr_1,Ytr.shoes);
    model.shoes_2 = fitcsvm(Xtr_2,Ytr.shoes);
else
    model.shoes = fitcsvm(Xtr,Ytr.shoes,'CrossVal',useCrossVal);
end

if strcmp(useCrossVal, 'off')
    if strcmp(model_name, "two")
        [l.shoes1,prob.shoes1] = predict(model.shoes_1,Xte_1);
        [l.shoes2,prob.shoes2] = predict(model.shoes_2,Xte_2);
        l.shoes = floor((l.shoes1+l.shoes2)/2);
        prob.shoes = max(prob.shoes1(:,2),prob.shoes2(:,2));
        prob.shoes = [-prob.shoes prob.shoes];
    else
        [l.shoes,prob.shoes] = predict(model.shoes,Xte);
    end
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
if strcmp(model_name, "two")
    model.upred_1 = fitcsvm(Xtr_1,Ytr.upred);
    model.upred_2 = fitcsvm(Xtr_2,Ytr.upred);
else
    model.upred = fitcsvm(Xtr,Ytr.upred,'CrossVal',useCrossVal);
end

if strcmp(useCrossVal, 'off')
    if strcmp(model_name, "two")
        [l.upred1,prob.upred1] = predict(model.upred_1,Xte_1);
        [l.upred2,prob.upred2] = predict(model.upred_2,Xte_2);
        l.upred = floor((l.upred1+l.upred2)/2);
        prob.upred = max(prob.upred1(:,2),prob.upred2(:,2));
        prob.upred = [-prob.upred prob.upred];
    else
        [l.upred,prob.upred] = predict(model.upred,Xte);
    end
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
% nPairs = length(data_idx); 
% visualise_attribute(te_img,prob,Yte,data_idx,nPairs, idx_attribute )
