function out = CC(ref,tar,mask)
%--------------------------------------------------------------------------
% Cross Correlation
%
% USAGE
%   out = CC(ref,tar,mask)
%
% INPUT
%   ref : reference HS data (rows,cols,bands)
%   tar : target HS data (rows,cols,bands)
%   mask: binary mask (rows,cols) (optional)
%
% OUTPUT
%   out : cross correlations (bands)
%
%--------------------------------------------------------------------------

if nargin==2
    [rows,cols,bands] = size(tar);

    out = zeros(1,bands);
    for i = 1:bands
        tar_tmp = tar(:,:,i);
        ref_tmp = ref(:,:,i);
        cc = corrcoef(tar_tmp(:),ref_tmp(:));
        out(1,i) = cc(1,2);
    end

else
    [rows,cols,bands] = size(tar);

    out = zeros(1,bands);
    mask = find(mask~=0);
    for i = 1:bands
        tar_tmp = tar(:,:,i);
        ref_tmp = ref(:,:,i);
        cc = corrcoef(tar_tmp(mask),ref_tmp(mask));
        out(1,i) = cc(1,2);
    end
end
 out=mean(out);   
