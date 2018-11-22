function [score ,bbox, allpeople, yolo_detections] = yolo_detect(gallery, image_size, thresh, hier_thresh)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    num_imgs = length(gallery);
    bbox = {};
    score = {};
    people = {};
    allpeople = {};
    yolo_detections = {};
    
    for j=1:num_imgs
    
        curr_img = gallery(j).image;  % imresize(gallery(j).image, image_size, 'bilinear');
        
        detections = yolomex('detect',curr_img,thresh,hier_thresh);
        yolo_detections{j} = detections;

        %% saving boxes
        result = [];
        score_tmp = [];
        ppl_images = {};
        num = size(detections,2);
        for i=1:num 
            p = detections(i);
            if strcmp(p.class, "person")
                score_tmp = [score_tmp; p.prob];
                x=floor((p.right-p.left)/2)+p.left;
                y=floor((p.bottom-p.top)/2)+p.top;
                if (p.bottom-p.top)/(p.right-p.left) > 128/64
                    dy = floor((p.bottom-p.top)/2);
                    dx = dy/2;
                else
                    dx = floor((p.right-p.left)/2);
                    dy = dx*2;
                end

                top=y-dy;
                bottom=y+dy;
                left=x-dx;
                right=x+dx;

                [imgheight, imgwidth, ~] = size(curr_img);

                if bottom > imgheight
                    bottom = imgheight;
                end
                if right > imgwidth
                    right = imgwidth;
                end
                if top <= 0
                    top = 1;
                end
                if left <= 0
                    left = 1;
                end

                img = curr_img(top:bottom,left:right,:);
                img = imresize(img,[128 64],'bilinear');
                ppl_images{length(ppl_images)+1} = img;
                allpeople{length(allpeople)+1} = img;
                result = [result; [left top right-left bottom-top]];
            end
        end
        score{j} = score_tmp;
        bbox{j} = result;
        people{j} = ppl_images;
    end
    
end

