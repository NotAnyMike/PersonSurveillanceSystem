function features_to_return = ExtractFeatureAttributeColor(images)

    dir = pwd();
    num = length(images);
    features_to_return = [];
    
    for i = 1:num
        
        fun = @(block_struct) histcounts(block_struct.data,5);
        
        img = images{i};
        %r = histcounts(img(:,:,1),5);
        %g = histcounts(img(:,:,2),5);
        %b = histcounts(img(:,:,3),5);
        win_size = [16 16];
        r = blockproc(img(:,:,1), win_size, fun);
        r = r(:)';
        g = blockproc(img(:,:,2), win_size, fun);
        g = g(:)';
        b = blockproc(img(:,:,3), win_size, fun);
        b = b(:)';
        t = histcounts(img, 5);
        t = t / sum(t);
        features = [r g b];
        features = features / sum(features);
        %features = [features t];
        features_to_return = [features_to_return;features]; 
    end