Q = zeros(1,2);
N = zeros(1,2);
e = 0.1;
avg = zeros(1,10000);

for i = 1:10000
    if(rand > e)
        [m,id] = max(Q);
        A = id;
    else
        temp = randperm(2);
        A = temp(1);
    end
    
    R = binaryBanditB(A);
    N(A) = N(A) + 1;
    Q(A) = Q(A) + (R - Q(A))/N(A);
    
    if i == 1
        avg(i) = R;
    else
        avg(i) = ((i-1)*avg(i-1) + R)/i;
    end
end

disp('Final Q-values:');
disp(Q);
disp('Maximum action count:');
disp(max(N));
disp('Action counts:');
disp(N);

figure
plot(1:10000, avg, "red");
ylim([0 1]);
title('Learning Curve (Running Average Reward)');
xlabel('Steps');
ylabel('Average Reward');
grid on;
