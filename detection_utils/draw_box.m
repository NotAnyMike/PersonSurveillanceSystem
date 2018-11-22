function draw_box(input,gt, bbox, detections)
% Xmin, Ymin, Xmax, Ymax 31, 326, 209, 712
%location = [31,209;326,712]; % x ; y in annotation
num = size(gt,1);
figure; clf;
imshow(input);hold on;
for i = 1:num
    tmp = gt(i,:);
    boundingbox = [tmp(1), tmp(2), tmp(3), tmp(4)];
    rectangle('Position',boundingbox,'EdgeColor','r','LineWidth',3);hold on;% x, y consistently
end 

for i = 1:size(bbox,1)
    tmp = bbox(i,:);
    boundingbox = [tmp(1), tmp(2), tmp(3), tmp(4)];
    rectangle('Position',boundingbox,'EdgeColor','g','LineWidth',3);hold on;% x, y consistently
end

% plot([[detections.left];[detections.right];[detections.right];[detections.left];[detections.left]],...
%      [[detections.top];[detections.top];[detections.bottom];[detections.bottom];[detections.top]],...
%      'LineWidth',3, 'Color','b');




%plot(location(1,1), location(2,1),'-g*');
