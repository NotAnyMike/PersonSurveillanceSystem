function train_data = ExtractFeatureAttribute(images)

    num = length(images);
    train_data = []; % to hold all the HoG for training
    
    for i = 1:num
        img = images{i};
        img = rgb2gray(img);
        %img = imresize(img,imsize,'bilinear');
        tmp = extractLBPFeatures(img,'CellSize',[8 8],'NumNeighbors',8);
        [frames, features] = vl_dsift(single(img),'Step',5,'size', 8,'fast');
        tmp = double(extractHOGFeatures(img,'CellSize',[16 16]));
        features = double(reshape(features,prod(size(features)),1)');
        tmp = tmp / max(tmp);
        features = features / max(features);
        train_data = [train_data;[tmp features]];
        %train_data = [train_data;tmp];
    end
    train_data = train_data;