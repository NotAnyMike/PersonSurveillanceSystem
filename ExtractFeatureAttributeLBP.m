function train_data = ExtractFeatureAttribute(images)

    num = length(images);
    train_data = []; % to hold all the HoG for training
    
    for i = 1:num
        img = images{i};
        img = rgb2gray(img);
        %img = imresize(img,imsize,'bilinear');
        tmp = extractLBPFeatures(img,'CellSize',[8 8],'NumNeighbors',8);
        tmp = tmp(:)';
        tmp = tmp / max(tmp);
        train_data = [train_data; tmp];
        %train_data = [train_data;tmp];
    end
    train_data = train_data;