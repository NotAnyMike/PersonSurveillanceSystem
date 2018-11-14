function [train_data,label] = ExtractFeatureAttribute(images, imsize, class)

    dir = pwd();
    num = length(images);
    features = []
    
    step_p = 5;
    binSize = 20;
    
    for i = 1:num
        img = images{i};
        img = rgb2gray(img);
        img = imresize(img,imsize,'bilinear');
        [frames, features] = vl_dsift(single(img),'Step',step_p,'size', binSize,'fast');
        %tmp = extractHOGFeatures(img,'CellSize',[16 16]);
        features = reshape(features,prod(size(features)),1);
        train_data = [train_data,features]; 
    end 
    
    num = size(train_data,1);
    if strcmp(class, 'pos')
        label = ones(num,1);
    end
    if strcmp(class, 'neg')
        label = zeros(num,1);
    end