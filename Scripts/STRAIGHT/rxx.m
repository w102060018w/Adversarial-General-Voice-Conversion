function R=rxx(X)%求自相关函数
N=length(X);
for m=1:N                     
    R(m)=0;
end
for m=1:N                      
    S=0; 
    for n=1:N+1-m
        H=X(n)*X(n+m-1);
        S=S+H;
    end
    R(m)=S/N;
end  