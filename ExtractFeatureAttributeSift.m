function train_data = ExtractFeatureAttribute(images, imsize)

    dir = pwd();
    num = length(images);
    features = [];
    train_data = [];
    
    step_p = 5;
    binSize = 4;
    
    for i = 1:num
        img = images{i};
        img = rgb2gray(img);
        img = imresize(img,imsize,'bilinear');
        [frames, features] = vl_dsift(single(img),'Step',step_p,'size', binSize,'fast');
        %tmp = extractHOGFeatures(img,'CellSize',[16 16]);
        features = reshape(features,prod(size(features)),1);
        train_data = [train_data,features]; 
    end