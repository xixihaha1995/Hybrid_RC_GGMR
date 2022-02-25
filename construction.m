function [x0] = construction()

% Goal:
% Get familiar with zone construction
% Get familair with IDF files
% Parameter details and calclulation of the nominal values.

%{
Function Name: construction.m
Input arguments:
    - None
Outputs:
    paranom: A struct of nominal values with two fields
        - rvalues: thermal conductance nominal values.
        - cvalues: thermal capacity nominal values.
    surfaces: A struct with two fields:
        - areas: calculated areas of different zone surfaces
        - sol_abs: coefficient of irradiance absorpiton or transmission, where applicable.
%}

%{

! Floor Area:        463.6 m2 (5000 ft2)
! Number of Stories: 1
!
! Zone Description Details:
!
!      (0,15.2,0)                      (30.5,15.2,0)
!           _____   ________                ____
!         |\     ***        ****************   /|
!         | \                                 / |
!         |  \                 (26.8,11.6,0) /  |
!         *   \_____________________________/   *
!         *    |(3.7,11.6,0)               |    *
!         *    |                           |    *
!         *    |                           |    *
!         *    |               (26.8,3.7,0)|    *
!         *    |___________________________|    *
!         *   / (3.7,3.7,0)                 \   *
!         |  /                               \  |
!         | /                                 \ |
!         |/___******************___***________\|
!          |       Overhang        |   |
!          |_______________________|   |   window/door = *
!                                  |___|
!
!      (0,0,0)                            (30.5,0,0)


%}

% Compute all required surface areas based on zone geometry from the IDF
% This has already been filled out for you, considering the trapizoidal
% shape of the zone. All areas are in sq meters.

%Aw = (13.7*1.2) ;           % window area m^2
%Add = (2.1*2.1) ;           % door area m^2
Aew = 63.6;    % back/outdoor external wall
%Ai32 = (5.16*2.4);          % internal wall b/w zone 3 & 2
%Ai34 = Ai32;                % internal wall b/w zone 3 & 4
%Ai35 = (23.1*2.4);          % internal wall b/w zone 3 & 5
%Ait = Ai32+Ai34+Ai35;       % total internal wall surface area
Af = 48;   % floor area (m2)
Ar = Af;                    % ceiling area = floor area
Awin = 12;


% For each lumped surface compute the thermal conductance and thermal capacity.
% The computed values will be used as the nominal parameter values for the
% on-linear paramter estimation. 


% ===== Back external wall ========
% The layers from outside to inside are indicated below:
%{

Outside Air Temp ---| C5 - 4 IN HW CONCRETE |--- Zone temperature

%}


% Outside layer material: wood siding
l_e1=0.009;      % !- Thickness {m}
k_e1=0.14;          % !- Conductivity {W/m-K}
d_e1=530;           % !- Density {kg/m3}
cp_e1=900;          % !- Specific Heat {J/kg-K}

% 2nd layer: fiberglass quilt
l_e2=0.066;      % !- Thickness {m}
k_e2=0.04;          % !- Conductivity {W/m-K}
d_e2=12;           % !- Density {kg/m3}
cp_e2=840;          % !- Specific Heat {J/kg-K}

% 3rd layer(inside layer): plasterboard
l_e3=0.012;      % !- Thickness {m}
k_e3=0.16;          % !- Conductivity {W/m-K}
d_e3=950;           % !- Density {kg/m3}
cp_e3=840;          % !- Specific Heat {J/kg-K}

% Compute the nominal value of the thermal conductance and capacitance
% . Write the expression in terms of the l_i,k_i,d_i, and cp_i variables. 
% the

Re = (l_e1/k_e1 +l_e2/k_e2+l_e3/k_e3)/Aew;
Ce = (d_e1*cp_e1*l_e1 + d_e2*cp_e2*l_e2 +d_e3*cp_e3*l_e3)*Aew;

% % ===== Interior walls ========
% % The layers from outside to inside are indicated below:
% %{
% 
% Neighboring zone(s) air temp ---| GP02 | AL21 | GP02 |--- Zone temperature
% 
% %}
% 
% % layer 1 material: GP02
% l_0=0.0159;      % !- Thickness {m}
% k_0=0.16;          % !- Conductivity {W/m-K}
% d_0=801;           % !- Density {kg/m3}
% cp_0=837;          % !- Specific Heat {J/kg-K}
% 
% % layer 2 material: AL21 (no themral mass)
% r_2=0.157;%               !- Thermal Resistance {m2-K/W}
% 
% % layer 3 same as layer 1
% 
% % Compute the nominal value of the thermal conductance and capacitance parameter.
% Ri = (l_0/k_0+r_2+l_0/k_0)/Af;
% Ci = (d_0*cp_0*l_0)*Ait;

% ===== Floor ========
% The layers from outside to inside are indicated below:
%{

Ground Temp ---| CC03 |--- Zone temperature

%}

% Outside layer material: insulation
l_f1=1.003;      % !- Thickness {m}
k_f1=0.04;          % !- Conductivity {W/m-K}
d_f1=0;           % !- Density {kg/m3}
cp_f1=0;          % !- Specific Heat {J/kg-K}

% 2nd layer material: timber flooring
l_f2=0.025;          % !- Thickness {m}
k_f2=0.14;           % !- Conductivity {W/m-K}
d_f2=650;           % !- Density {kg/m3}
cp_f2=1200;          % !- Specific Heat {J/kg-K}

% Compute the nominal value of the thermal conductance and capacitance parameter.
Rf = (l_f1/k_f1 + l_f2/k_f2)/Af;
Cf = (d_f1*cp_f1*l_f1 + d_f2*cp_f2*l_f2)*Af;
    
% ===== Ceiling ========
% The ceiling surface of the zone is not connected to the roof directly.
% There is a return air plenum in between. We can consider the return
% plenum as anoter zone in itself and model the interaction with the plenum
% directly. This helps simply the zone modeling further. The thermal
% conductance for the celing layer is given below.

% Outside layer material: Roofdeck
l_r1=0.019;      % !- Thickness {m}
k_r1=0.14;          % !- Conductivity {W/m-K}
d_r1=530;           % !- Density {kg/m3}
cp_r1=900;          % !- Specific Heat {J/kg-K}

% 2nd layer material: Fiberglass quilt
l_r2=0.1118;      % !- Thickness {m}
k_r2=0.04;          % !- Conductivity {W/m-K}
d_r2=12;           % !- Density {kg/m3}
cp_r2=840;          % !- Specific Heat {J/kg-K}

% 3rd layer material (inside): Plasterboard
l_r3=0.01;      % !- Thickness {m}
k_r3=0.16;          % !- Conductivity {W/m-K}
d_r3=950;           % !- Density {kg/m3}
cp_r3=840;          % !- Specific Heat {J/kg-K}
% Compute the nominal value of the thermal conductance and capacitance
% . Write the expression in terms of the l_i,k_i,d_i, and cp_i variables. 
% the

Rc = (l_r1/k_r1 +l_r2/k_r2+l_r3/k_r3)/Ar;
Cc = (d_r1*cp_r1*l_r1+d_r2*cp_r2*l_r2+d_r3*cp_r3*l_r3)*Ar;

% % ===== Window and door ========
% % Double layers surfaces. Each layer is the material: CLEAR 3mm 
% 
% l_0=0;              % !- Thickness {m}
% k_0=0;                % !- Conductivity {W/m-K}
% 
% % Compute the nominal value of the thermal conductance parameter (door and
% % window combined into one parameter.)
Rw = 0.4765/Awin;
% rd = 1/5.894/Add;
% Rw = Rw+rd;

% % ===== Air properties ========

Roair_ew = 11.9;             % m2K/W Outside air thermal resistance.
Roair_r = 14.4;             % m2K/W Outside air thermal resistance.
Roair_f =0.8;             % m2K/W Outside air thermal resistance.
Riair_ew = 2.2;          % m2K/W Internal air thermal resistance.
Riair_r = 1.8;          % m2K/W Internal air thermal resistance.
Riair_f = 2.2;          % m2K/W Internal air thermal resistance.
cpair = 1006;           % J/kg-K Specific heat
dair = 1.292498;           % kg/m3 Density
vair = 129.6;         % m3 Volumne
Cair = dair*cpair*vair; % J/K Thermal capacitance of the zone air.

Rim_1=Re; % internal mass
Rim_2=Re; % internal mass
Cim_1=Ce; % internal mass
Cim_2=Ce; % internal mass

theta1=1.0;% qgc
theta2=1.0;% qsol,tr
theta3=1.0;% qinf
% Construct a structure with the nominal values of all the parameters:
x0=[1/(Aew*Roair_ew) Re 1/(Aew*Riair_ew) 1/(Ar*Roair_r) ...
    Rc 1/(Af*Riair_r) Rf 1/(Af*Riair_ew) Rw Rim_1 Rim_2...
    Ce/2 Ce/2 Cc/2 Cc/2 Cf  Cair Cim_1 Cim_2 theta1 theta2 theta3];
%x0=[0.000975324975466503	0.00974044982712022	0.000712803518506621	0.0228435806803563	0.711997612628489	1.00090100886670	0.0175393623091634	0.00257949661551930	2.65484790477176	3.19328032829068	0.394222324862232	4570965.97329418	187251.667614230	49172917.9512013	178743766.940200	2163698.52950222	28838.9375787144	844814346.265025	9596.90567155649	2.08367207563203	1.95006422889737	0.00117659005274230];
%x0=[1.31953792864864,0.00890510366853912,0.000628855403945960,0.0273569022253588,0.0568420676579543,0.000324307341489654,0.000528039536943624,9.46966858488324,0.000997605647856287,122103.100503121,2151.89058971868,791825.402789919,9194848.53273205,455553.952956955,175.344431078262]';
% paranom = struct('rvalues',[],'cvalues',[]);
% % rvalues (K/W)
% paranom.rvalues(1) = 1/(Aew*Roair_ew);
% paranom.rvalues(2) = Re;                
% paranom.rvalues(3) = 1/(Aew*Riair_ew);
          
% paranom.rvalues(4) = 1/(Ar*Roair_r);
% paranom.rvalues(5) = Rc;                
% paranom.rvalues(6) = 1/(Af*Riair_r);
% paranom.rvalues(7) = Rf;
% paranom.rvalues(8) = 1/(Af*Riair_ew);
% paranom.rvalues(9) = Rw;  
% % add/edit the paranom rvalues list as required by your model. 
% 
% % cvalues (J/K)
% paranom.cvalues(1) = Ce/2;
% paranom.cvalues(2) = Ce/2;
% %paranom.cvalues(3) = Ci;            
% paranom.cvalues(3) = Cc/2;
% paranom.cvalues(4) = Cc/2;
% paranom.cvalues(5) = Cf; 
% paranom.cvalues(6) = Cair;
% add/edit the paranom cvalues list as required by your model. 

end

















    

