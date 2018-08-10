%% Rotation Information for HAPT Dataset
%--------------------------------------------------------------------------
% Last Updated: 8/10/2018
% Description: This script plots the rotation data for the HAPT data set as
% an approximation from the gyroscope information.  We use the cross points
% from GUPR to break pieces of the rotation sequence into distinct regions
% depending on the crosses.

%% Import Data (Must Run Cross Points First in Python)
%--------------------------------------------------------------------------
Root_train = "/Volumes/Seagate Backup Plus Drive/UnifyID/HAPT/UCI HAR Dataset/train/Inertial Signals/";
Root_test = "/Volumes/Seagate Backup Plus Drive/UnifyID/HAPT/UCI HAR Dataset/test/Inertial Signals/";

% Training Data
f_xTrain = fopen(strcat(Root_train, "body_gyro_x_train.txt"));
f_yTrain = fopen(strcat(Root_train, "body_gyro_y_train.txt"));
f_zTrain = fopen(strcat(Root_train, "body_gyro_z_train.txt"));
cross_train = csvread(strcat(Root_train, "/crossings/_train_cross_times.csv"));

dx_train = textscan(f_xTrain, '%f'); dy_train = textscan(f_yTrain, '%f'); dz_train = textscan(f_zTrain, '%f'); 
x_train = reshape(dx_train{1,1}, [128 7352])'; y_train = reshape(dy_train{1,1}, [128 7352])'; z_train = reshape(dz_train{1,1}, [128 7352])';

% Test Data
f_xTest = fopen(strcat(Root_test, "body_gyro_x_test.txt"));
f_yTest = fopen(strcat(Root_test, "body_gyro_y_test.txt"));
f_zTest = fopen(strcat(Root_test, "body_gyro_z_test.txt"));
cross_test = csvread(strcat(Root_test, "/crossings/_test_cross_times.csv"));

dx_test = textscan(f_xTest, '%f'); dy_test = textscan(f_yTest, '%f'); dz_test = textscan(f_zTest, '%f'); 
x_test = reshape(dx_test{1,1}, [128 2947])'; y_test = reshape(dy_test{1,1}, [128 2947])'; z_test = reshape(dz_test{1,1}, [128 2947])';

fclose('all');

clear f_xTrain f_yTrain f_zTrain dx_train dy_train dz_train f_xTest f_yTest f_zTest dx_test dy_test dz_test

%% Convert Times to Indices 
t = 0:0.02:2.54; % Standard length of time.

[row_train, col_train] = size(cross_train);
[row_test, col_test]   = size(cross_test);

Cross_Train = zeros(row_train, col_train);
Cross_Test  = zeros(row_test, col_test);

for i = 1:row_train
    for j = 1:col_train
        if cross_train(i,j) ~= 0
            Cross_Train(i,j) = find(cross_train(i,j) < t, 1, 'first');
        end
    end
end

for i = 1:row_test
    for j = 1:col_test
        if cross_test(i,j) ~= 0
            Cross_Test(i,j) = find(cross_test(i,j) < t, 1, 'first');
        end
    end
end

%% Convert Gyro to Rotations
%--------------------------------------------------------------------------
L_train = 7352; L_test = 2947; L_time = 128;
Rots = zeros(3,3,L_time);
Rots_Train = {};
Rots_Test  = {};

for i = 1:L_train
    fprintf(strcat("Processing file", " ", int2str(i), " ", "of", " ", int2str(L_train), "\n")) 
    Rots = Gyro_2_Rots(x_train(i,:), y_train(i,:), z_train(i,:));
    Rots_Train{i,1} = Rots;
    stops = Cross_Train(i,:); stops = stops(stops ~= 0);
    Tripods_Spacing(Rots, 50, stops);
    Path = fullfile(char(strcat(Root_train, "crossings")), char(strcat("gyro_train_ind=", int2str(i-1),"_R")));
    saveas(gcf, Path, 'epsc')
    close all
end

for i = 1:L_test
    fprintf(strcat("Processing file", " ", int2str(i), " ", "of", " ", int2str(L_test), "\n")) 
    Rots = Gyro_2_Rots(x_test(i,:), y_test(i,:), z_test(i,:));
    Rots_Test{i,1} = Rots;
    stops = Cross_Test(i,:); stops = stops(stops ~= 0);
    Tripods_Spacing(Rots, 50, stops);
    Path = fullfile(char(strcat(Root_test, "crossings")), char(strcat("gyro_test_ind=", int2str(i-1),"_R")));
    saveas(gcf, Path, 'epsc')
    close all
end



