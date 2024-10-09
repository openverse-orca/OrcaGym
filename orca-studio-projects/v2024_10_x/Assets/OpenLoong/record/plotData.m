clear variables; close all
dataRec=load('datalog.log');
simTime=dataRec(:,1:1);
motor_pos_des=dataRec(:,2:32);
motor_pos_cur=dataRec(:,33:63);
motor_vel_des=dataRec(:,64:94);
motor_vel_cur=dataRec(:,95:125);
motor_tor_des=dataRec(:,126:156);
rpyVal=dataRec(:,157:159);
base_omega_W=dataRec(:,160:162);
gpsVal=dataRec(:,163:165);
base_vel=dataRec(:,166:168);
dX_cal=dataRec(:,169:180);
Ufe=dataRec(:,181:192);
Xd=dataRec(:,193:204);
X_cur=dataRec(:,205:216);



figure();
for i = 1:14
    subplot(3,5,i);
    plot(simTime, motor_pos_cur(:,i),'b-',...
         simTime, motor_pos_des(:,i),'r-')
end
figure();
for i = 1:2
    subplot(2,3,i);
    plot(simTime, motor_pos_cur(:,i+14),'b-',...
         simTime, motor_pos_des(:,i+14),'r-')
end
for i = 1:3
    subplot(2,3,i+3);
    plot(simTime, motor_pos_cur(:,i+16),'b-',...
         simTime, motor_pos_des(:,i+16),'r-')
end
figure();
for i = 1:12
    subplot(3,4,i);
    plot(simTime, motor_pos_cur(:,i+19),'b-',...
         simTime, motor_pos_des(:,i+19),'r-')
end

% figure();
% for i = 1:12
%     subplot(3,4,i);
%     plot(simTime, motor_tor_out(:,i+19),'b-');
% end

figure("Name","base");
for i = 1:3
    subplot(2,3,i);
    plot(simTime,gpsVal(:,i));
    subplot(2,3,i+3);
    plot(simTime,base_vel(:,i));    
end

figure("Name","dX_cal");
for i = 1:12
    subplot(4,3,i);
    plot(simTime, dX_cal(:,i));    
end

figure("Name","Ufe");
for i = 1:12
    subplot(4,3,i);
    plot(simTime, Ufe(:,i));    
end

figure("Name","Xd");
for i = 1:12
    subplot(4,3,i);
    plot(simTime, X_cur(:,i),simTime, Xd(:,i));    
end


