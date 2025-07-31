clear all

Dir = fullfile('Pictures\list3_1');
allImages = imageDatastore(Dir);
% load the calibration data for calibrating the images
%load('CalPics_ZoomCam\ZcamParams.mat')

% Display images to be stitched.
figure;
montage(allImages.Files)

order = false; % enable this for image ordering
%% code for ordering the images based on features and transformation matrix
if order

    numImages = numel(allImages.Files);
    features = cell(numImages,0);
    vpoints = cell(numImages,0);
    direction = cell(numImages);
    inliersNo = zeros(numImages);
    for i= 1:numImages
        I = readimage(allImages,i);
        grayImage = im2gray(I);
        points = detectSURFFeatures(grayImage, 'NumOctaves', 8);
        [features{i}, vpoints{i}] = extractFeatures(grayImage,points,"Upright",true);
    end
    
    for i= 1:numImages
        for j = i+1:numImages
            % ii
            % jj
            indexPairs = matchFeatures(features{i}, features{j}, 'Unique', true);
            matchedPoints = vpoints{i}(indexPairs(:,1), :);
            matchedPoints2 = vpoints{j}(indexPairs(:,2), :);
            [tform, inliers, status] = estimateGeometricTransform2D(matchedPoints, matchedPoints2,  'projective', ...
                            'Confidence', 99.9, 'MaxNumTrials', 2000, ...
                            'MaxDistance', 1.50);
            inliersNo(i,j) = size(inliers,1);
            direction{i,j} = tform.T(3,1:2);
        end
    end
    inliersNoC = inliersNo'+inliersNo;
    directionC = direction;
    for inx = 1:numImages
        for iny = inx+1:numImages
            directionC{iny,inx} = -1.*direction{inx,iny};
        end
    end
    panoPos = zeros(numImages+numImages-1);
    panoPos(numImages,numImages) = 1;
    
    Io = 1;
    placed = zeros(numImages,0);
    placed(1) = 1;
    A = 1:numImages;
    for ind = 2:numImages
        [~,I] = max(inliersNoC(Io,:));
    
        if ismember(I,placed)
            for inda = 2:numImages
                if ~ismember(inda,placed)
                    Io = inda;
                    break;
                end
            end
            continue;
        end
    
        [~,II] = max(abs(directionC{Io,I}));
        m = directionC{Io,I}(II);
        [Ior,Ioc] = find(panoPos==Io);
        if II == 1
            panoPos(Ior,Ioc-sign(m)) = I;
        else
            panoPos(Ior-sign(m),Ioc) = I;
        end
        placed(ind) = I;
        inliersNoC(Io,I) = 0;
        inliersNoC(I,Io) = 0;
        Io = I;
    end

allImages.Files = allImages.Files(placed);

figure;
montage(allImages.Files)

end
%% 

% Read the first image from the image set.
I = readimage(allImages,1);
% undistort the images if needed
%I = undistortImage(I,ZcamParams);

% find features for the first image
grayImage = im2gray(I);
points = detectSURFFeatures(grayImage);
[features, points] = extractFeatures(grayImage,points,"Upright",true);

% Initialize the transformations array, use affine or projective based on
% the images
numImages = numel(allImages.Files);
tforms(numImages) = affinetform2d; %projtform2d
imageSize = zeros(numImages,2);

% Extract features of the remaining images
for n = 2:numImages
    % Store points and features for I(n-1).
    pointsPrevious = points;
    featuresPrevious = features;
    
    % Read I(n).
    I = readimage(allImages, n);
    
    %I = undistortImage(I,ZcamParams);
    
    % Convert image to grayscale.
    grayImage = im2gray(I);    
    
    % Save image size.
    imageSize(n,:) = size(grayImage);
    
    % Detect and extract Upright-SURF features for I(n).
    points = detectSURFFeatures(grayImage,'NumOctaves', 8);    
    [features, points] = extractFeatures(grayImage, points, "Upright",true);
  
    % Find correspondences between I(n) and I(n-1).
    indexPairs = matchFeatures(features, featuresPrevious, 'Unique', true);
    %matches(n) = size(indexPairs,1);
    matchedPoints = points(indexPairs(:,1), :);
    matchedPointsPrev = pointsPrevious(indexPairs(:,2), :);        
    
    % Estimate the transformation between I(n) and I(n-1).
    [tforms(n),inlierIdx] = estgeotform2d(matchedPoints, matchedPointsPrev,...
        'affine', 'Confidence', 99.9, 'MaxNumTrials', 2000, 'MaxDistance', 1.50); %projective
    matches(n) = size(inlierIdx,1);
    % Compute T(1) * T(2) * ... * T(n-1) * T(n).
    tforms(n).A = tforms(n-1).A * tforms(n).A; 
end

%% Code For Smoothly Blending the Images Together

% h = 201;
% w = 103;
% c = 3;
% wx = ones(1, w);
% wx(1:ceil(w/3)) = linspace(0, 1, ceil(w/3));
% wx(floor(2*w/3 + 1):w) = linspace(1, 0, ceil(w/3));
% wx = repmat(wx, h, 1, c);
% wy = ones(h, 1);
% wy(1:ceil(h/3)) = linspace(0, 1, ceil(h/3));
% wy(floor(2*h/3 + 1):h) = linspace(1, 0, ceil(h/3));
% wy = repmat(wy, 1, w, c);
% weight = wx .* wy;
% imshow(weight)
%%

% Compute the output limits for each transformation.
for i = 1:numel(tforms)           
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);    
end

avgXLim = mean(xlim, 2);
[~,idx] = sort(avgXLim);
centerIdx = floor((numel(tforms)+1)/2);
centerImageIdx = idx(centerIdx);

Tinv = invert(tforms(centerImageIdx));
for i = 1:numel(tforms)    
    tforms(i).A = Tinv.A * tforms(i).A;
end

for i = 1:numel(tforms)           
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);
end

maxImageSize = max(imageSize);

% Find the minimum and maximum output limits. 
xMin = min([1; xlim(:)]);
xMax = max([maxImageSize(2); xlim(:)]);

yMin = min([1; ylim(:)]);
yMax = max([maxImageSize(1); ylim(:)]);

% Width and height of panorama.
width  = round(xMax - xMin);
height = round(yMax - yMin);

% Initialize the "empty" panorama.
panorama = ones([height width 3], 'like', I);

blender = vision.AlphaBlender('Operation', 'Binary Mask','MaskSource', 'Input port');  

% Create a 2-D spatial reference object defining the size of the panorama.
xLimits = [xMin xMax];
yLimits = [yMin yMax];
panoramaView = imref2d([height width], xLimits, yLimits);

% Create the panorama.
for i = 1:numImages
    
    I = readimage(allImages, i);   
    
    %I = undistortImage(I,ZcamParams);
    
    % Transform I into the panorama.
    warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);
                  
    % Generate a binary mask.    
    mask = imwarp(true(size(I,1),size(I,2)), tforms(i), 'OutputView', panoramaView);
    
    % Overlay the warpedImage onto the panorama.
    panorama = step(blender, panorama, warpedImage, mask);
end

figure
imshow(panorama)

%% image cropping code
% source Github link: https://github.com/preethamam

stitchedImage = panorama;
% Initilize the variables
w = size(stitchedImage,2);
h = size(stitchedImage,1);

% Convert the stitched image to grayscale and threshold it
% such that all pixels greater than zero are set to 255
% (foreground) while all others remain 0 (background)
gray = rgb2gray(stitchedImage);
% if strcmp(input.canvas_color,'black')
    BW = imbinarize(gray, 1/255);%input.blackRange
% else
%     BW = imbinarize(gray, input.whiteRange/255);
%     BW = imcomplement(BW);
% end

% Find all external contours in the threshold image then find
% the *largest* contour which will be the contour/outline of
% the stitched image
BW2 = imfill(BW, 'holes');

% Canvas outer indices
canvas_outer_indices = BW2 == 0;

% Normalize the image to -1 and others
stitched = double(stitchedImage);
stitched(repmat(canvas_outer_indices,1,1,3)) = -255;
stitched = stitched / 255.0;

% Get the crop indices
maxarea = 0;
height  = zeros(1,w);
left    = zeros(1,w); 
right   = zeros(1,w);
        
ll = 0;
rr = 0;
hh = 0; 
nl = 0;

for line = 1:h
    for k = 1:w
        p = stitched(line,k,:);
        m = max(max(p(1), p(2)), p(3));
        if m < 0 
            height(k) =  0; 
        else 
            height(k) = height(k) + 1; % find Color::NO
        end
    end
        
    for k = 1:w
        left(k) = k;            
        while (left(k) > 1 && (height(k) <= height(left(k) - 1)))
            left(k) = left(left(k) - 1);
        end
    end
            
    for k = w - 1:-1:1
        right(k) = k;
        while (right(k) < w - 1 && (height(k) <= height(right(k) + 1)))
            right(k) = right(right(k) + 1);
        end
    end
            
    for k = 1:w
        val = (right(k) - left(k) + 1) * height(k);
        if(maxarea < val)
            maxarea = val;
            ll = left(k); 
            rr = right(k);
            hh = height(k); 
            nl = line;
        end
    end
end

% Crop indexes
cropH = hh + 1;
cropW = rr - ll + 1;
offsetx = ll;
offsety = nl - hh + 1;

% Cropped image
croppedImage = stitchedImage(offsety : offsety + cropH, offsetx : offsetx + cropW,:);  

%show the cropped image
imshow(croppedImage);
%%