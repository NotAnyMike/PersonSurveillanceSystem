function extracted_features = ExtractFeatureReid(images, settings)
    %
    % Setting should look like:
    dsettings = struct;
    % color histogram
    dsettings.color_hist.use       = true;
    dsettings.color_hist.nbins      = 5;
    dsettings.color_hist.win_size  = [16 16];
    % hsv histogram
    dsettings.col_hist_hsv.use      = true;
    dsettings.col_hist_hsv.nbins    = 5;
    dsettings.col_hist_hsv.win_size = [16 16];
    % Sift Features
    dsettings.sift.use     = true;
    dsettings.sift.step_p  = 5;
    dsettings.sift.binSize = 8;
    %lbp
    dsettings.lbp.use         = true;
    dsettings.lbp.win_size    = [8 8];
    dsettings.lbp.n_neighbour = 8;
    dsettings.lbp.radius      = 1;
    dsettings.lbp.is_upright  = true;
    %BoW with Sift Features
    dsettings.bow_with_sift.use          = false;
    dsettings.bow_with_sift.step         = 3;
    dsettings.bow_with_sift.binSize      = 3;
    dsettings.bow_with_sift.num_of_words = 200;
    dsettings.bow_with_sift.vocab        = 0;
    %BoW with HoG features
    dsettings.bow_with_hog.use          = true;
    dsettings.bow_with_hog.num_of_words = 200;
    dsettings.bow_with_hog.NumBins      = 18;
    dsettings.bow_with_hog.BlockSize    = [2 2];
    dsettings.bow_with_hog.CellSize     = [8 8];
    dsettings.bow_with_hog.num_blocks   = 3;
    dsettings.bow_with_hog.vocab        = 0;

    
    
    dir = pwd();
    num = length(images);
    extracted_features = [];
    
    for i = 1:num
        img      = images{i};
        img_gray = rgb2gray(img);
        tmp = [];
        
        % Color histograms
        if settings.color_hist.use
            fun = @(block_struct) histcounts(block_struct.data, settings.color_hist.nbins);
            r = blockproc(img(:,:,1), settings.color_hist.win_size, fun);
            r = r(:)';
            g = blockproc(img(:,:,2), settings.color_hist.win_size, fun);
            g = g(:)';
            b = blockproc(img(:,:,3), settings.color_hist.win_size, fun);
            b = b(:)';
            color_features = [r g b];
            color_features = color_features / sum(color_features);
            tmp = [tmp, color_features];
        end
        
        if (settings.col_hist_hsv.use) 
            hsv_fun = @(block_struct) histcounts(block_struct.data, settings.col_hist_hsv.nbins);
            img_hsv = rgb2hsv(img);
            h = blockproc(img_hsv(:,:,1), settings.col_hist_hsv.win_size, hsv_fun);
            h = h(:)';
            s = blockproc(img_hsv(:,:,2), settings.col_hist_hsv.win_size, hsv_fun);
            s = s(:)';
            v = blockproc(img_hsv(:,:,3), settings.col_hist_hsv.win_size, hsv_fun);
            v = v(:)';
            t = histcounts(img_hsv, settings.col_hist_hsv.nbins);
            hsv_features = [h s t];
            hsv_features = hsv_features / sum(hsv_features);
            tmp = [tmp, hsv_features];
        end
        
        if settings.sift.use
            [frames, sift_features] = vl_dsift(single(img_gray),'Step',settings.sift.step_p,'size', settings.sift.binSize,'fast');
            %sift_features = reshape(sift_features,prod(size(sift_features)),1);
            sift_features = sift_features(:)';
            tmp = [tmp , sift_features]; 
        end
        
        if settings.lbp.use
            lgp_features = extractLBPFeatures(img_gray,'CellSize', settings.lbp.win_size,'NumNeighbors',settings.lbp.n_neighbour,...
                'Radius', settings.lbp.radius, 'Upright', settings.lbp.is_upright);
            tmp = [tmp, lgp_features];
        end
        
        if settings.bow_with_sift.use
            img_smooth = vl_imsmooth(single(img_gray), 0.1);
            [frames, features] = vl_dsift(single(img_smooth),'Step',settings.bow_with_sift.step,'size', settings.bow_with_sift.binSize,'fast');
            d = vl_alldist2(double(settings.bow_with_sift.vocab), double(features));
            [~,min_index] = min(d);
            bowsift_features = histcounts(min_index,size(settings.bow_with_sift.vocab,2));
            bowsift_features = bowsift_features / sum(bowsift_features);
            tmp = [tmp, bowsift_features];
        end
        
        if settings.bow_with_hog.use
            [features, visualization] = extractHOGFeatures(img_gray,'CellSize',settings.bow_with_hog.CellSize, 'BlockSize', settings.bow_with_hog.BlockSize, 'NumBins', settings.bow_with_hog.NumBins);
            num_blocks2 = int16(size(features,2)/settings.bow_with_hog.NumBins);
            features = reshape(features,settings.bow_with_hog.NumBins, num_blocks2);
            d = vl_alldist2(double(settings.bow_with_hog.vocab), double(features));
            [~,min_index] = min(d);
            bowhog_features = hist(min_index,size(settings.bow_with_hog.vocab,2));
            tmp = [tmp, bowhog_features];
        end
        
        extracted_features = [extracted_features; tmp];
        
    end 