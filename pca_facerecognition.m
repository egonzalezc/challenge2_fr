
global sizeImg;
sizeImg = [70,160];
numEigenvectors = 10;

% call training
trainSet = readData('training/');
[eigenvectors, meanFace] = training(trainSet, numEigenvectors);

% call test
testSet = readData('test/');
threshold = 20;
for i=1:size(testSet,2)
    testing(test_faces(:,i), eigenvectors, meanFace, threshold); 
end

%%
%PCA
function [eigenvectors, meanFace] = training(imagesMatrix, numEigenvectors)    
    meanFace=mean(imagesMatrix,2); 
    normImages = imagesMatrix - meanFace;
    figure;
    imshow(getRGBImage(meanFace));
    
    %Compute eigenvectors and eigenvalues using a smaller matrix than the conv
    smallMatrix=normImages'*normImages;
    [eigenvectors,eigenvalues]=eig(smallMatrix);
    eigenvectors= normImages*eigenvectors;
    eigenvectors = normc(eigenvectors);
    
    %sort eigen values in descending order so the first K eigenvectors will
    %be those with the most relevant information
    [value, column] = sort(diag(eigenvalues), 'descend');
    eigenvectors = eigenvectors(:,column);

    displayEigenFaces(eigenvectors, numEigenvectors);
 
    %Select the first 10 eigenvectors
    eigenvectors = eigenvectors(:,1:numEigenvectors); 
  
end

%% Test eigenmouth/ Detect or no a mouth in the given input vector
function [] = testing(inputImgVector, eigenvectors, meanFace, threshold)
    global sizeImg;
    normInput = inputImgVector - meanFace;
    
    %project on the eigenspace
    weights = (eigenvectors' * normInput);
    projection =  eigenvectors * weights;
    
    distance = norm(normInput - projection);
    distanceScore = 1/(1+distance);
%     if distance < threshold
%             fprintf( 'It is a mouth with a distance of %f score %f', distance, distanceScore);
%         else
%             fprintf( 'It is not a mouth');
%     end  
    img = getRGBImage(projection);
    imwrite(uint8(img), sprintf('results/Projection_%f.jpg',distance));
    imwrite(reshape(inputImgVector, [sizeImg,3]), sprintf('results/Distance=%f.jpg',distance));
end

%% Tools
function [mouths] =  readData(path)
    global sizeImg;
    global nimages;
    
    folderName = strcat(path,'*.jpg');
    imagefiles = dir(folderName);
    nimages = length(imagefiles);
    mouths = zeros(prod([sizeImg,3]), nimages);
    for i=1:nimages
       filename = strcat(path, imagefiles(i).name);
       currentImage = im2double(imread(filename));
       currentImage = imresize(currentImage,sizeImg);
       vectorImage = reshape(currentImage, 1, []);
       mouths(:, i) = vectorImage;
    end
end

function displayEigenFaces(eigenVectors, numEigenvectors)
    global sizeImg;
    rows =  prod([sizeImg,3]);

     minvalue = min(eigenVectors);
     eigenVectors = eigenVectors-repmat(minvalue, [rows, 1]);
     maxvalue = max(eigenVectors);

     eigenVectors = 255* eigenVectors./repmat(maxvalue, [rows, 1]);
    
    figure;
    colormap hsv;
    for i = 1:numEigenvectors
        subplot(5,2,i);
        v = reshape(eigenVectors(:,i),[sizeImg,3]);
        imshow(uint8(v));
        title(sprintf('Eigenface %d',i));
    end
end


function [imgRGB] = getRGBImage(vectorImage)
    global sizeImg;
    minvalue = min(vectorImage);
    vectorImage = vectorImage-minvalue;
    maxvalue = max(vectorImage);
    vectorImage = 255* vectorImage./maxvalue;
    imgRGB = uint8(reshape(vectorImage,[sizeImg,3]));
end