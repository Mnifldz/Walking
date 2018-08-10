function Rots = Gyro_2_Rots(x,y,z)
%% Convert Gyroscope to Rotations
%--------------------------------------------------------------------------
% Last Updated: 8/10/2018
% Description: Converts gyroscope data to rotations.  We assume that Rots
% is nxnxL where L is the length of the time series of nxn rotations.
% x,y,z give the angular velocity in each frame and times gives the time
% stamp.  The function is written particularly for n=3

L = length(x);
Rots = zeros(3,3,L); Rots(:,:,1) = eye(3);

for i = 2:L
    Lie_Alg = [0, x(i-1), -y(i-1); -x(i-1), 0, z(i-1); y(i-1), -z(i-1), 0];
    theta = norm([x(i-1), y(i-1), z(i-1)], 2);
    Rots(:,:,i) = eye(3) + (sin(theta)/theta)*Lie_Alg + ((1-cos(theta))/theta^2)*(Lie_Alg)^2;
end