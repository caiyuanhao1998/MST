function s=csnr(A,B,row,col)

[n,m,ch]=size(A);
summa = 0;
if ch==1
   e=A-B;
   e=e(row+1:n-row,col+1:m-col);
   me=mean(mean(e.^2));
   s=10*log10(255^2/me);
else
    for i=1:ch
        e=A-B;
        e=e(row+1:n-row,col+1:m-col,i);
        mse = mean(mean(e.^2));
        s  = 10*log10(255^2/mse);
        summa = summa + s;
    end
        s = summa/ch;
end

return;

