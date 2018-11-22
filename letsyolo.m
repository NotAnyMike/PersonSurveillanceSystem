function letsyolo()
%LETSYOLO Summary of this function goes here
%   Detailed explanation goes here

    cd yolomex

    datacfg = fullfile(pwd,'darknet/cfg/coco.data');
    cfgfile = fullfile(pwd,'yolov2.cfg');
    weightfile = fullfile(pwd,'yolov2.weights');

    yolomex('init',datacfg,cfgfile,weightfile);

    cd ..

end

