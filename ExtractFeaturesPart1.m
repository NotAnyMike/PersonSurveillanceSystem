function [tmp] = ExtractFeaturesPart1(img, extract_settings)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

img_grey = rgb2gray(img);
tmp = [];  % empty placeholder array for features

if (extract_settings.col_hist_rgb.use)
    correlogram_features = blockproc(img, extract_settings.correlogram.window, extract_settings.correlogram.fun);
    tmp = [tmp, correlogram_features(:)'];
end

if (extract_settings.col_hist_rgb.use)
    r = blockproc(img(:,:,1), extract_settings.col_hist_rgb.window, extract_settings.col_hist_rgb.fun);
    r = r(:)';
    g = blockproc(img(:,:,2), extract_settings.col_hist_rgb.window, extract_settings.col_hist_rgb.fun);
    g = g(:)';
    b = blockproc(img(:,:,3), extract_settings.col_hist_rgb.window, extract_settings.col_hist_rgb.fun);
    b = b(:)';
    colour_features = [r g b];
    colour_features = colour_features / sum(colour_features);
    tmp = [tmp, colour_features];
end

if (extract_settings.col_hist_hsv.use)
    img_hsv = rgb2hsv(img);
    h = blockproc(img_hsv(:,:,1), extract_settings.col_hist_hsv.window, extract_settings.col_hist_hsv.h_fun);
    h = h(:)';
    s = blockproc(img_hsv(:,:,2), extract_settings.col_hist_hsv.window, extract_settings.col_hist_hsv.s_fun);
    s = s(:)';
    v = blockproc(img_hsv(:,:,3), extract_settings.col_hist_hsv.window, extract_settings.col_hist_hsv.v_fun);
    v = v(:)';
    t = histcounts(img_hsv, extract_settings.col_hist_hsv.nbin);
    colour_features = [h s t];
    colour_features = colour_features / sum(colour_features);
    tmp = [tmp, colour_features];
end

if (extract_settings.hog.use)
    hog_features = extractHOGFeatures(img_grey,'CellSize',extract_settings.hog.window,...
        'NumBins', extract_settings.hog.nbins, 'BlockSize', extract_settings.hog.block_size);
    tmp = [tmp, hog_features];
end

if (extract_settings.lbp.use)
    lgp_features = extractLBPFeatures(img_grey,'CellSize',extract_settings.lbp.window,'NumNeighbors',extract_settings.lbp.n_neighbour,...
        'Radius', extract_settings.lbp.radius, 'Upright', extract_settings.lbp.is_upright);
    tmp = [tmp, lgp_features];
end

if (extract_settings.sift.use)
    img_grey = imresize(img_grey, resize_size, 'bilinear');
    [~, sift_features] = vl_dsift(single(img_grey),'Step',extract_settings.sift.step,'size', extract_settings.sift.nbin,'fast');
    tmp = [tmp, sift_features(:)];
end

if (extract_settings.mscr.use)
    [mvec,pvec] = detect_mscr_masked(im2double(img),extract_settings.mscr.mask, extract_settings.mscr.p);
    tmp = [tmp, mvec(:)', pvec(:)'];
end    

end
