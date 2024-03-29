function train_data = ExtractFeatureReid(images, imsize, vocabulary, NumBins,BlockSize,CellSize,num_blocks)
    
    using_bow = true;
    dir = pwd();
    num = length(images);
    train_data = []; % to hold all the HoG for training
    
    step_p  = 30;
    binSize = 20;
    
    if exist('vocabulary','var') == 0
        fprintf('No vocabulary found, constructing one');
        vocabulary = codebook(images, 400);
    end
    
    for i = 1:num
        img = images{i};
        img = rgb2gray(img);
        img = imresize(img,imsize,'bilinear');
        
        if using_bow
            [features, visualization] = extractHOGFeatures(img,'CellSize',CellSize, 'BlockSize', BlockSize, 'NumBins', NumBins);
            num_blocks2 = int16(size(features,2)/NumBins);
            features = reshape(features,NumBins, num_blocks2);
            d = vl_alldist2(double(vocabulary), double(features));
            [~,min_index] = min(d);
            tmp = hist(min_index,size(vocabulary,2));
        else
            [~, tmp] = vl_dsift(single(img),'Step',step_p,'size', binSize,'fast');
        end
        
        train_data = [train_data;tmp]; 
    end 