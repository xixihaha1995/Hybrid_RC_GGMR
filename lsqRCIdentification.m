%% least square function to estimate Rs and Cs
function [x]=lsqRCIdentification(x0)
%x = lsqnonlin(@CostFunc,x0,lb,ub);
%x = nlinfit(u,ym,@CostFunc,x0,lb,ub);
%[x,resnorm,residual,exitflag,output] = lsqcurvefit(@Func,x0,u,ym,lb,ub);%
lb=0.01*x0;
ub=100*x0;
nvars=length(x0);
%x = lsqnonlin(@Func,x0,lb,ub);
options = optimoptions(@particleswarm,'SwarmSize',500,'MaxIterations',150,'PlotFcn','pswplotbestf','Display','iter');
options.InitialSwarmMatrix = x0;
x1 = particleswarm(@FuncRMSE,nvars,lb,ub,options);
x2=x1;
lb=0.1*x1;
ub=10*x1;
options.InitialSwarmMatrix = x2;
x = particleswarm(@FuncRMSE,nvars,lb,ub,options);

% option2 = optimoptions(@lsqnonlin,'Algorithm','levenberg-marquardt','MaxFunctionEvaluations',1500);
% x = lsqnonlin(@Funclsq,x1,lb,ub,option2);
end
function F = FuncRMSE(x)
load nonlinear.mat;
yt = SS_function(x,utrain,ymtrain,n,m,ts);
F=(sum((yt-ymtrain).^2));
end

