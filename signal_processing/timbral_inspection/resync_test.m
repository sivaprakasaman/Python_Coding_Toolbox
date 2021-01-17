%Andrew Sivaprakasam
%Purdue University
%Email: asivapr@purdue.edu

%Just code to verify my python function, lol. 
%Still building confidence...

addpath 'instruments'
[x,fs] = audioread('violin_A4_normal.mp3');
 
f_0 = 440;
t_vect = (1:length(x))/fs;
    
% x = (1/fs)*cos(2*pi*440*t_vect)';
% x = x + (1/fs)*cos(2*pi*880*t_vect)';
 
f_vect = [1:10]*f_0;

comb = f_vect'*t_vect;

w = dpss(numel(t_vect),1,1);
w = w/sum(w);

x_sin = x'.*w'.*sin(2*pi*comb);
x_cos = x'.*w'.*cos(2*pi*comb);

sin_sum = sum(x_sin,2);
cos_sum = sum(x_cos,2);

mags = sqrt(sin_sum.^2+cos_sum.^2);
stem(f_vect,mags/max(mags));