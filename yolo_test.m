addpath(genpath('./yolomex'))

cd yolomex

datacfg = fullfile(pwd,'darknet/cfg/coco.data');
cfgfile = fullfile(pwd,'yolov2.cfg');
weightfile = fullfile(pwd,'yolov2.weights');
% filename = fullfile(pwd,'darknet/data/person.jpg');
filename = fullfile(pwd, 'test.jpg');
thresh = 0.24;
hier_thresh = 0.5;
I = imread(filename);

yolomex('init',datacfg,cfgfile,weightfile);
cd ..
load('./data/people_search/people_search.mat');
num_imgs = length(gallery);
bbox = {};
score = {};
gallery_image_size = [540, 960];




for j=1:num_imgs
    
    curr_img = imresize(gallery(j).image, gallery_image_size, 'bilinear');
    
    %detections = yolomex('test',filename,thresh,hier_thresh);   
    detections = yolomex('detect',curr_img,thresh,hier_thresh);%yolomex('detect',I,thresh,hier_thresh);    

    %% saving boxes

    result = [];
    score_tmp = [];
    images = {};
    num = size(detections,2);
    for i=1:num 
        p = detections(i);
        if strcmp(p.class, "person")
            score_tmp = [score_tmp; p.prob]
            x=floor((p.right-p.left)/2)+p.left;
            y=floor((p.bottom-p.top)/2)+p.top;
            if (p.bottom-p.top)/(p.right-p.left) > 128/64
                dy = floor((p.bottom-p.top)/2);
                dx = dy/2;
            else
                dx = floor((p.right-p.left)/2);
                dy = dx*2;
            end

            top=y+dy;
            bottom=y-dy;
            left=x-dx;
            right=x+dx;

            [imgtop, imgright, ~] = size(I);

            if top > imgtop
                top = imgtop;
            end
            if right > imgright
                right = imgright;
            end
            if bottom < 0
                bottom = 0;
            end
            if left < 0
                left = 0;
            end

            img = I(bottom:top,left:right,:);
            img = imresize(img,[128 64],'bilinear');
            images{length(images)+1} = img;
            result = [result; [top left right-left top-bottom]];
            %imshow(img);
        end
    end
    score{j} = score_tmp;
    bbox{j} = result;
end

%% getting all imagest to print at the same time
imgs = [];
for i=1:length(images)
    imgs = [imgs, images{i}];
end
f1=figure;
imshow(imgs);
f2=figure;
plotResults();