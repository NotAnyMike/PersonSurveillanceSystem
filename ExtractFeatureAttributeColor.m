function train_data = ExtractFeatureAttributeColor(images)

    dir = pwd();
    num = length(images);
    train_data = [];
    
    for i = 1:num
        img = images{i};
        r = histcounts(img(:,:,1),5);
        g = histcounts(img(:,:,2),5);
        b = histcounts(img(:,:,3),5);
        features = [r g b];
        features = features / max(features);
        train_data = [train_data;features]; 
    end