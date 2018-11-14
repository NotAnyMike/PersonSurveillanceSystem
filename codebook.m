
function dictionary = codebook(images,num_of_words,NumBins,BlockSize,CellSize,num_blocks)
fprintf('\nvocabulary\n');
num = size(images,1);
container = [];
[height,width,~] = size(images{1});

using_hog = true;

for i = 1:num
    img = images{i};
    img = rgb2gray(img); % ignoring channels
    img = vl_imsmooth(single(img), 0.5);
    
    %[~, features] = vl_dsift(single(input_img),'Step',step_p,'size', binSize,'fast');
    %use vl_hog
    %features = vl_phow(img,'fast','true');
    
    % If using HoG
    if using_hog
        [features,visualization]  = extractHOGFeatures(img,'CellSize',CellSize, 'BlockSize', BlockSize, 'NumBins', NumBins);
        num_blocks2 = int16(size(features,2)/NumBins);
        features = reshape(features, NumBins, num_blocks2);
        container = [container,features];
    else
        [~, features] = vl_dsift(single(input_img),'Step',step_p,'size', binSize,'fast');
        container = [container,(single(features'))];
    end
        
    if mod(i,60) == 0
        fprintf('image %d/%d\n',i,num);
    end
end

fprintf('\nstart to building vocabulary')
dictionary = vl_kmeans(container,num_of_words);
fprintf('\nfinish building vocabulary')

