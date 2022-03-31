function y = SS_function(x,um,ym,n,m,ts)
%load nonlinear.mat;
%x=[R_e1 R_e2 R_e3 C_e1 C_e2 R_c1 R_c2 R_c3 C_c1 C_c2 R_f1 R_f2 R_f3 C_f1 C_f2];
[A B C D]=M_ssMatrix(x,n,m);
%n=6;
%ntrain=4380;
y=zeros(length(ym),1);% initialization
xinter=25+zeros(n,1);
% Create a MATLAB state-space model object.
%sys = ss('Pass the correct arguments to this function');
sys=ss(A,B,C,D);
% Model discretization.
sysd = c2d(sys,ts);
Ad = sysd.A;    Bd = sysd.B;    
Cd = sysd.C;    Dd = sysd.D;
% Use the discrete state space parameters to predict the model response for
% each training data sample. Remember, all we need to predict Y(i) is the
% intial state x(0) and the inputs from u(0) till u(i). 
Yall=zeros(length(ym),1);
for i = 1:length(ym)
    %xinter = 'Simulate the model states';
    %Yall(i,:) = 'Simulate the model output';
    Yall(i,:) = Cd*xinter+Dd*um(:,i);
	xinter = Ad*xinter + Bd*um(:,i); 
end
y = Yall(:,end);