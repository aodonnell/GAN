%% main.m 
% Created by Alex O'Donnell and Pat Savoie
% Date: November 21st, 2017
%
% Directions: follow through the sections. Do not 
%
% Implements the following...
% Sunghbae's DCGAN from: https://github.com/sunghbae/dcgan-matconvnet
% matconvnet-1.0-beta24 from: http://www.vlfeat.org/matconvnet/

clear 
clc

addpath('src');
addpath('net');

%% 1) Install and Compile MatConvNet. Will not work on machines with no C/C++ SDK
fprintf('Installing and compiling MatConvNet library. This may take a few minutes.\n')
install.matconvnet_path = 'matconvnet-1.0-beta24/matlab/vl_setupnn.m';
untar('http://www.vlfeat.org/matconvnet/download/matconvnet-1.0-beta24.tar.gz') ;
run(install.matconvnet_path) ;

vl_compilenn('enableGpu', false); % change this to true if you want to run with GPU

fprintf('Finished install. Matconvnet is ready!\n')

%% 2) Copy customized parameters of network into matconvnet. 
% Running this section overwrites files related to optimization
% in the default matconvnet. Sets up adam optimizer and rms probagation
% used both in the generator and the discriminator.

src = 'src/matlab/*';
dst = 'matconvnet-1.0-beta24/matlab/';
copyfile(src, dst);

src = 'src/+solver/*';
dst = 'matconvnet-1.0-beta24/examples/+solver';
copyfile(src, dst);

fprintf('Custom parameters ready.\n')

%% 3) Test dcgan with a pre-trained model in order to generate new images or...
% will save images to the amount stored in "test.num_images". Figures
% generated will be montages of 64 64x64 pixel images with 

mkdir('generated_montages')
test.save_img_path = 'generated_montages'; % path to the output images 
test.num_images = 1; % num of images to be generated
test.idx_gpus = 0; % change to 1 for GPU

get_test_DCGAN(test);

fprintf([num2str(test.num_images) ' generated image(s) can be found in ' test.save_img_path '.\n'])

%% 4) Here you can train the GAN yourself
% WARNING: will update the model stored in 'net'.
% WARNING: this may take several days to train. It took 3 days to get a
% useable model on my macbook pro.

fprintf(['Program is paused. Press any key to start training. Note that this\n'...
        'will overwrite any existing model (including the pre-trained model)\n'...
        'Training for a matter of days may be required in order to achieve\n' ...
        'similar performance. Press Control + C to abort.\n']);
pause;

train.matconvnet_path = 'matconvnet-1.0-beta24/matlab/vl_setupnn.m';
train.idx_gpus = 0;

get_train_DCGAN(train);

fprintf('Train dcgan.... Done \n')