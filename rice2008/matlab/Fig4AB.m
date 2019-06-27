
[init_states,state_names] = rice_model_2008_init_states();

%set SL to 2.2
SL_ind = find(strcmp(state_names, 'SL'))
init_states(SL_ind) = 2.2

[params,param_names] = rice_model_2008_init_parameters();
start_time_ind = find(strcmp(state_names, 'start_time'))
Ca_diastolic_ind = find(strcmp(state_names, 'Ca_diastolic'))
SLmin_ind = find(strcmp(state_names, 'SLmin'))

params(start_time_ind) = 1000;
params(Ca_diastolic_ind) = 10;
params(SLmin_ind) = 2.5;


length(init_states)

[t1,s1] = ode15s(@rice_model_2008_rhs, [0,650],init_states,[],params);

plot(t1,s1(:,SLind))

%SL_ind = state_names('SL') 
%rice.state_indices("SL")
%p = (rice.init_parameter_values(start_time=1000,Ca_diastolic=10,SLmin=2.5),)
%s1 = odeint(rice.rhs,init,t1,p)
%pylab.plot(t1,s1[:,SL_ind])

%release against different afterload values:
% al = np.linspace(0,0.8,9)
% v = []
% intf_ind = rice.state_indices('intf')
% init = s1[-1,:]
% init[intf_ind] = 0.0
% t2 = np.linspace(650,1000,351)
% for load in al:
%     p = (rice.init_parameter_values(start_time=1000,Ca_diastolic=10,SEon=0,fixed_afterload=load),)
%     s2 = odeint(rice.rhs,init,t2,p)
%     v.append(-1000*(s2[50,SL_ind]-init[SL_ind])/50)
%     pylab.plot(t2,s2[:,SL_ind],label=str(load))

%set correct axes for figure 1:
% pylab.ylim([1.4,2.3])
% pylab.ylabel("SL (micrometers)")
% pylab.xlim([550,1000])
% pylab.xlabel("time (ms)")
% pylab.legend()

%compute fitted Hill velocity:
% f_hill = np.linspace(0,0.8,101)
% a_hill = 0.16
% b_hill = 2.4
% v_max = 14
% v_hill = (v_max*a_hill-b_hill*f_hill)/(f_hill+a_hill)

% pylab.figure(2)
% pylab.plot(al,v,label="Model")
% pylab.plot(f_hill,v_hill,label="Hill curve")
% pylab.ylabel("v (micrometers/s")
% pylab.xlabel("normalized force")
% 
% pylab.show()



