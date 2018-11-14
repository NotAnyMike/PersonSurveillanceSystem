function codebook = codebook_sift(images, num_of_words, step_p, binSize)

    num = length(images);
    features = [];
    
    for i = 1:num
        img = images{i};
        img = rgb2gray(img);
        img = vl_imsmooth(single(img), 0.1);
        [frames, ind_features] = vl_dsift(single(img),'Step',step_p,'size', binSize,'fast');
        features = [features, ind_features];
    end
    
    [codebook,~] = vl_kmeans(single(features),num_of_words);