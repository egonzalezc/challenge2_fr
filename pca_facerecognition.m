
global sizeImg;
sizeImg = [70,160];
neigen_faces = 10; %Quantity of eigenvectors to keep

%read training and tests data
training_faces = readData('training/');
test_faces = readData('test/');

% call training
[eigen_vectors, meanface] = training(training_faces, neigen_faces);

% call test
threshold = 20;
for i=1:size(test_faces,2)
    testing(test_faces(:,i), eigen_vectors, meanface, threshold); 
end

%%
%PCA
function [eigen_vectors, meanfaces] = training(faces, neigen_faces)    
    %Calculate the mean
    meanfaces=mean(faces,2); 
    normalized_faces = faces - meanfaces;

%     displayImage(meanfaces);
    
    %Calculate the covariance matrix.
%     covfaces=faces*faces';
    L=normalized_faces'*normalized_faces;

    %Calculate the eigenvalues and eigenvectors.
%     [eigen_vectors,eigen_values]=eig(covfaces);
     eigen_vectors= normalized_faces*L;
     eigen_vectors = normc(eigen_vectors);
    
     %sort eigen values in descending order so the first K eigenvectors will
    %be those with the most relevant information
%     [value, column] = sort(diag(eigen_values), 'descend');
%     eigen_vectors = eigen_vectors(:,column);

    displayEigenFaces(eigen_vectors, neigen_faces);
 
    %Select the first 10 eigenvectors
    eigen_vectors = eigen_vectors(:,1:neigen_faces); 

    
end

%% Test eigenmouth/ Detect or no a mouth in the given input vector
function [] = testing(input_vector, eigen_vectors, meanface, threshold)
    global sizeImg;
    normalized_input = input_vector - meanface;
    %project on the eigenspace
    weights = (eigen_vectors' * normalized_input);
    projection =  eigen_vectors * weights;
    
    img = displayImage(projection);
    
    distance = norm(normalized_input - projection);
    distance_score = 1/(1+distance);
%     if distance < threshold
%             fprintf( 'It is a mouth with a distance of %f score %f', distance, distance_score);
%         else
%             fprintf( 'It is not a mouth');
%     end  
    imwrite(uint8(img), sprintf('results/Projection_%f.jpg',distance));
    imwrite(reshape(input_vector, [sizeImg,3]), sprintf('results/Original_%f.jpg',distance));
end

%% Tools
function [faces] =  readData(path)
    global sizeImg;
    global nimages;
    
    folder_name = strcat(path,'*.jpg');
    imagefiles = dir(folder_name);
    nimages = length(imagefiles);
    faces = zeros(prod([sizeImg,3]), nimages);
    for i=1:nimages
       filename = strcat(path, imagefiles(i).name);
       currentimage = im2double(imread(filename));
       currentimage = imresize(currentimage,sizeImg);
       vectorimage = reshape(currentimage, 1, []);
       faces(:, i) = vectorimage; %Matrix of images vector
    end
end

function displayEigenFaces(eigen_vectors, neigen_faces)
    global sizeImg;
    rows =  prod([sizeImg,3]);

     minvalue = min(eigen_vectors);
     eigen_vectors = eigen_vectors-repmat(minvalue, [rows, 1]);
     maxvalue = max(eigen_vectors);

     eigen_vectors = 255* eigen_vectors./repmat(maxvalue, [rows, 1]);
    
    figure;
    colormap hsv;
    for i = 1:neigen_faces
        subplot(5,2,i);
        v = reshape(eigen_vectors(:,i),[sizeImg,3]);
        imshow(uint8(v));
        title(sprintf('Eigenface %d',i));
    end
end

function [img] = displayImage(vector_image)
    global sizeImg;
    minvalue = min(vector_image);
    vector_image = vector_image-minvalue;
    maxvalue = max(vector_image);
    vector_image = 255* vector_image./maxvalue;

%     figure;
    colormap hsv;
    img = reshape(vector_image,[sizeImg,3]);
%     imshow(uint8(img));   
end