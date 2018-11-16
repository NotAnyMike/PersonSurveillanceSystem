function train_data = ExtractFeatureAttributeBoWSift(images, vocabulary, step_p, binSize)

    num = length(images);
    train_data = [];

    for i = 1:num
        img = images{i};
        img = rgb2gray(img);
        img = vl_imsmooth(single(img), 0.1);
        [frames, features] = vl_dsift(single(img),'Step',step_p,'size', binSize,'fast');
        
        d = vl_alldist2(double(vocabulary), double(features));
        [~,min_index] = min(d);
        tmp = histcounts(min_index,size(vocabulary,2));
        tmp = tmp / sum(tmp);
        
        train_data = [train_data; tmp]; 
    end 