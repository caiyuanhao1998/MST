%% plot color pics
clear; clc;
method_name = {'desci','gapnet', 'gaptv', 'HSSP', 'lamnet', 'TSA', 'twist','truth','dgsmp'};
for i=1:9
    load(['simulation_results\',method_name{i},'.mat']);
end
res = permute(res,[1,3,4,2]);

load(['simulation_results\','mst','.mat']);
pred_block_mst = pred;

lam28 = [453.5 457.5 462.0 466.0 471.5 476.5 481.5 487.0 492.5 498.0 504.0 510.0...
    516.0 522.5 529.5 536.5 544.0 551.5 558.5 567.5 575.5 584.5 594.5 604.0...
    614.5 625.0 636.5 648.0];

pred_block_desci(find(pred_block_desci>0.7))=0.7;
pred_block_gapnet(find(pred_block_gapnet>0.7))=0.7;
pred_block_gaptv(find(pred_block_gaptv>0.7))=0.7;
pred_block_HSSP(find(pred_block_HSSP>0.7))=0.7;
pred_block_lamnet(find(pred_block_lamnet>0.7))=0.7;
pred_block_TSA(find(pred_block_TSA>0.7))=0.7;
pred_block_twist(find(pred_block_twist>0.7))=0.7;
truth(find(truth>0.7))=0.7;
res(find(res>0.7))=0.7;
pred_block_twist(find(pred_block_mst>0.7))=0.7;

f = 2;

%% plot spectrum
figure(123);
[yx, rect2crop]=imcrop(sum(squeeze(truth(f, :, :, :)), 3));
rect2crop=round(rect2crop)
close(123);

figure; 
spec_mean_desci = mean(mean(squeeze(pred_block_desci(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_gapnet= mean(mean(squeeze(pred_block_gapnet(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_gaptv = mean(mean(squeeze(pred_block_gaptv(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_HSSP = mean(mean(squeeze(pred_block_HSSP(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_lamnet = mean(mean(squeeze(pred_block_lamnet(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_TSA = mean(mean(squeeze(pred_block_TSA(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_twist = mean(mean(squeeze(pred_block_twist(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_truth = mean(mean(squeeze(truth(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_dgsmp = mean(mean(squeeze(res(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_mst = mean(mean(squeeze(pred_block_mst(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);

spec_mean_desci = spec_mean_desci./max(spec_mean_desci);
spec_mean_gapnet = spec_mean_gapnet./max(spec_mean_gapnet);
spec_mean_gaptv = spec_mean_gaptv./max(spec_mean_gaptv);
spec_mean_HSSP = spec_mean_HSSP./max(spec_mean_HSSP);
spec_mean_lamnet = spec_mean_lamnet./max(spec_mean_lamnet);
spec_mean_TSA = spec_mean_TSA./max(spec_mean_TSA);
spec_mean_twist = spec_mean_twist./max(spec_mean_twist);
spec_mean_truth = spec_mean_truth./max(spec_mean_truth);
spec_mean_dgsmp = spec_mean_dgsmp./max(spec_mean_dgsmp);
spec_mean_mst = spec_mean_mst./max(spec_mean_mst);

corr_desci = roundn(corr(spec_mean_truth(:),spec_mean_desci(:)),-4);
corr_gapnet = roundn(corr(spec_mean_truth(:),spec_mean_gapnet(:)),-4);
corr_gaptv = roundn(corr(spec_mean_truth(:),spec_mean_gaptv(:)),-4);
corr_HSSP = roundn(corr(spec_mean_truth(:),spec_mean_HSSP(:)),-4);
corr_lamnet = roundn(corr(spec_mean_truth(:),spec_mean_lamnet(:)),-4);
corr_TSA = roundn(corr(spec_mean_truth(:),spec_mean_TSA(:)),-4);
corr_twist = roundn(corr(spec_mean_truth(:),spec_mean_twist(:)),-4);
corr_dgsmp = roundn(corr(spec_mean_truth(:),spec_mean_dgsmp(:)),-4);
corr_mst = roundn(corr(spec_mean_truth(:),spec_mean_mst(:)),-4);

% legends = strings(9, 1);

X = lam28;

Y(1,:) = spec_mean_truth(:); 
Y(2,:) = spec_mean_desci(:); Corr(1)=corr_desci;
Y(3,:) = spec_mean_gaptv(:); Corr(2)=corr_gaptv;
Y(4,:) = spec_mean_HSSP(:); Corr(3)=corr_HSSP;
Y(5,:) = spec_mean_lamnet(:); Corr(4)=corr_lamnet;
Y(6,:) = spec_mean_TSA(:); Corr(5)=corr_TSA;
Y(7,:) = spec_mean_twist(:); Corr(6)=corr_twist;
Y(8,:) = spec_mean_gapnet(:); Corr(7)=corr_gapnet;
Y(9,:) = spec_mean_dgsmp(:); Corr(8)=corr_dgsmp;
Y(10,:) = spec_mean_mst(:); Corr(9)=corr_mst;

createfigure(X,Y,Corr)


