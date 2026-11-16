clear;clc;
res_path = fullfile('results', 'Test_result.mat');
load(res_path);

metrics = zeros(size(pred, 1), 6);
psnr_total=0.0;
ssim_total=0.0;
psnr_list = zeros(size(pred, 1),1);
ssim_list = zeros(size(pred, 1),1);
for i=1:size(pred, 1)
    Z = squeeze(pred(i,:,:,:));
    Z = double(Z);
    S = squeeze(truth(i,:,:,:));
    S = double(S);
    
    Z(Z>1.0) = 1.0;
    Z(Z<0.0) = 0.0;
    
    [psnr, rmse, ergas, sam, uiqi, ssim] = quality_assessment(double(im2uint8(S)), double(im2uint8(Z)), 0, 1);
    
    metrics(i,1) = psnr;
    metrics(i,2) = rmse;
    metrics(i,3) = ergas;
    metrics(i,4) = sam;
    metrics(i,5) = uiqi;
    metrics(i,6) = ssim;
    psnr_list(i,1) = psnr;
    ssim_list(i,1) = ssim;
    
    psnr_total = psnr_total+psnr;
    ssim_total = ssim_total+ssim;
end
psnr = mean(psnr_list);
ssim = mean(ssim_list);
fprintf('The PNSR=%f\n',psnr);
fprintf('The SSIM=%f\n',ssim);


save(fullfile('results', 'quality_metrics.mat'), 'metrics', 'psnr_list', 'ssim_list');
