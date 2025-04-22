
%% Haralick Texture Feature Analysis 
%%
%Some default parameter
numRows = 1;
mumCols = 1;

% Define the directory you want to list files from
directory = '/Users/jackxia/Desktop/Matlab/Test folder for Haralick'; % Change this to your directory path
% Get a list of all files in the directory
files = dir(directory);
% Define the file name
filename = 'output.csv';
% Open the file for writing
fid = fopen(filename, 'w');
% Check if the file is opened successfully
if fid == -1
    error('Cannot open file: %s', filename);
end

% Loop through each file and print its name
for i = 1:length(files)

    % Check if it's not a directory
    if ~files(i).isdir
        currentName = files(i).name;
        [~, ~, ext] = fileparts(currentName);
        if strcmp(ext, '.mat')
            % Load the .mat file
            load(currentName);

            %calculate feature values
            AUC_t = extractHaralickFeatures(data.AUC.*data.tumormask, numRows, mumCols);
            TTP_t = extractHaralickFeatures(data.TTP.*data.tumormask, numRows, mumCols);
            peakI_t = extractHaralickFeatures(data.peakI.*data.tumormask, numRows, mumCols);
            %decorr_t = extractHaralickFeatures(data.dtime.*data.tumormask, numRows, mumCols);

            %write them to csv
            fprintf(fid, '%s\n', currentName);
            fprintf(fid, '%s\n', "AUC Tumor ROI");
            fprintf(fid, '%f,%f,%f,%f\n', AUC_t);
            fprintf(fid, '%s\n', "TTP Tumor ROI");
            fprintf(fid, '%f,%f,%f,%f\n', TTP_t);
            fprintf(fid, '%s\n', "peakI Tumor ROI");
            fprintf(fid, '%f,%f,%f,%f\n', peakI_t);
            % fprintf(fid, '%s\n', "decorr Tumor ROI");
            % fprintf(fid, '%f,%f,%f,%f\n', decorr_t);


            %Same process but for healthy tissue
            AUC_h = extractHaralickFeatures(data.AUC.*data.healthymask, numRows, mumCols);
            TTP_h = extractHaralickFeatures(data.TTP.*data.healthymask, numRows, mumCols);
            peakI_h = extractHaralickFeatures(data.peakI.*data.healthymask, numRows, mumCols);
            %decorr_h = extractHaralickFeatures(data.dtime.*data.healthymask, numRows, mumCols);

            %write them to csv
            fprintf(fid, '%s\n', currentName);
            fprintf(fid, '%s\n', "AUC healthy ROI");
            fprintf(fid, '%f,%f,%f,%f\n', AUC_h);
            fprintf(fid, '%s\n', "TTP healthy ROI");
            fprintf(fid, '%f,%f,%f,%f\n', TTP_h);
            fprintf(fid, '%s\n', "peakI healthy ROI");
            fprintf(fid, '%f,%f,%f,%f\n', peakI_h);
            % fprintf(fid, '%s\n', "decorr healthy ROI");
            % fprintf(fid, '%f,%f,%f,%f\n', decorr_h);

            clear data
        else
            warning('Skipping non-mat file: %s', currentName);
        end
        
    end
end

%%
function haralickFeatures = extractHaralickFeatures(imageData, numRows, numCols)
    % EXTRACTHARALICKFEATURES Extract Haralick features from subplots of an image.
    %
    % haralickFeatures = EXTRACTHARALICKFEATURES(imageData, numRows, numCols)
    % extracts Haralick features from the given imageData divided into numRows
    % and numCols subplots. The output is a matrix of Haralick features.

    % Calculate the dimensions of each subplot
    subplotHeight = size(imageData, 1) / numRows;
    subplotWidth = size(imageData, 2) / numCols;

    % Preallocate an array to store the Haralick features
    haralickFeatures = zeros(numRows * numCols, 4);

    % Loop over each subplot and extract Haralick features
    index = 1;
    for row = 1:numRows
        for col = 1:numCols
            % Extract the current subplot
            subplot = imageData(1 + (row - 1) * subplotHeight:row * subplotHeight, ...
                                1 + (col - 1) * subplotWidth:col * subplotWidth, :);
            % Convert the subplot to grayscale
            graySubplot = mat2gray(subplot);
            % Calculate the gray-level co-occurrence matrix (GLCM)
            glcm = graycomatrix(graySubplot);
            % Calculate the Haralick features
            stats = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
            % Store the features in the array
            haralickFeatures(index, :) = [stats.Contrast, stats.Correlation, stats.Energy, stats.Homogeneity];
            index = index + 1;
        end
    end
end
