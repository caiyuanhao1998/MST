function [] = dispCubeAshwin(datacube,brightness,mywl,labelss,cols,rows,writefile,grayc,resultname)
%dispCubeAshwin(x_twist16,80,linspace(450,650,23),[],4,6,0,0)
h = figure;
set(gcf,'color','white');
% fig_xpos = 10;  % 50
% fig_ypos = -30;  % 50
% fig_width = 820; % 820
% fig_height = 860; % 860
% set(gcf,'position',[fig_xpos fig_ypos fig_width fig_height]);

% wltextposition = [0 size(datacube,2)-0];
wltextposition = [12 12]; %[12 40];
m=size(datacube,3);
for ind=1:m
    datacube(:,:,ind)=flipud(datacube(:,:,ind)); % up side dowm
end
imagedatacube2(h,double(datacube),brightness,mywl,labelss,wltextposition,cols,rows,writefile,grayc,resultname);




function imagedatacube2(h, data,mymax,wavelengths,labelss,textposition,cols,rows,writefile,grayc,resultname)
%data = datacube to plot
%mymax = scaling factor to increase brightness of each slice
%wavelengths = wavelengths of each spectral slice in nm
%textposition = where the wavelengths text will be placed

n = size(data,1);
% data=mymax*data;
temp=sort(data(:));
temp=mean(temp(end-50:end));
data=data*(50/temp);
if mymax > 1
    data=data*mymax;
end

totalfigs = size(data,3);

% cols = ceil(sqrt(totalfigs));
% rows = ceil(totalfigs/cols);

figscount = 1;
set(gcf,'color','white');

subplot1(rows,cols,'Gap',[0.005 0.005]);
for r = 1:rows
    for c = 1:cols
        if figscount>totalfigs
            subplot1(figscount);axis off;
        else
            currentwl = wavelengths(figscount);

            cmM=(gray*kron(ones(3,1),spectrumRGB(wavelengths(figscount))));
            cmM=cmM/max(max(cmM));

            if figscount<=totalfigs
                subplot1((r-1)*cols+c);
                if isempty(labelss)
                    subimage2(squeeze(data(:,:,figscount)),colormap(cmM));
                    %text(textposition(1),textposition(2),['\bf' num2str(currentwl,4) ' nm'],'color','w')
                else
                    if grayc==1
                        subimage2(make01(squeeze(data(:,:,figscount))));
                    else
                        subimage2(squeeze(data(:,:,figscount)),colormap(cmM));
                    end
                    if r*c==1
                        text(textposition(1),textposition(2),['\bf' num2str(labelss(r*c),4) ' frame'],'color','w')
                    else
                        text(textposition(1),textposition(2),['\bf' num2str(labelss((r-1)*cols+c),4) ' frames'],'color','w')
                    end
                end
                if writefile
                    imwrite(squeeze(rot90(rot90(data(:,:,figscount)))),colormap(cmM),['file' num2str(figscount) '.png'])
                end
                axis off;
            end
        end
        figscount = figscount+1;
    end
end
% h
% rect1 = [181 37 553 553];
% h_crop = imcrop(h,rect1);
% saveas(h_crop, [resultname '.png']); 
saveas(h, [resultname, '.png']);
% print(gcf,'-dpng',resultname);


function sRGB = spectrumRGB(lambdaIn, varargin)
%spectrumRGB   Converts a spectral wavelength to RGB.
%
%    sRGB = spectrumRGB(LAMBDA) converts the spectral color with wavelength
%    LAMBDA in the visible light spectrum to its RGB values in the sRGB
%    color space.
%
%    sRGB = spectrumRGB(LAMBDA, MATCH) converts LAMBDA to sRGB using the
%    color matching function MATCH, a string.  See colorMatchFcn for a list
%    of supported matching functions.
%
%    Note: LAMBDA must be a scalar value or a vector of wavelengths.
%
%    See also colorMatchFcn, createSpectrum, spectrumLabel.

%    Copyright 1993-2005 The MathWorks, Inc.

if (numel(lambdaIn) ~= length(lambdaIn))
    
    error('spectrumRGB:unsupportedDimension', ...
          'Input must be a scalar or vector of wavelengths.')
    
end

% Get the color matching functions.
if (nargin == 1)
    
    matchingFcn = '1964_full';
    
elseif (nargin == 2)
    
    matchingFcn = varargin{1};
    
else
    
    error(nargchk(1, 2, nargin, 'struct'))
    
end

[lambdaMatch, xFcn, yFcn, zFcn] = colorMatchFcn(matchingFcn);

% Interpolate the input wavelength in the color matching functions.
XYZ = interp1(lambdaMatch', [xFcn; yFcn; zFcn]', lambdaIn, 'pchip', 0);

% Reshape interpolated values to match standard image representation.
if (numel(lambdaIn) > 1)
    
    XYZ = permute(XYZ', [3 2 1]);
    
end

% Convert the XYZ values to sRGB.
XYZ2sRGB = makecform('xyz2srgb');
sRGB = applycform(XYZ, XYZ2sRGB);


function hout = subimage2(varargin)
%SUBIMAGE Display multiple images in single figure.
%   You can use SUBIMAGE in conjunction with SUBPLOT to create
%   figures with multiple images, even if the images have
%   different colormaps. SUBIMAGE works by converting images to
%   truecolor for display purposes, thus avoiding colormap
%   conflicts.
%
%   SUBIMAGE(X,MAP) displays the indexed image X with colormap
%   MAP in the current axes.
%
%   SUBIMAGE(I) displays the intensity image I in the current
%   axes.
%
%   SUBIMAGE(BW) displays the binary image BW in the current
%   axes.
%
%   SUBIMAGE(RGB) displays the truecolor image RGB in the current
%   axes.
%
%   SUBIMAGE(x,y,...) displays an image with nondefault spatial
%   coordinates.
%
%   H = SUBIMAGE(...) returns a handle to the image object.
%
%   Class Support
%   -------------
%   The input image can be of class logical, uint8, uint16,
%   or double.
%
%   Example
%   -------
%       load trees
%       [X2,map2] = imread('forest.tif');
%       subplot(1,2,1), subimage(X,map)
%       subplot(1,2,2), subimage(X2,map2)
%
%   See also IMSHOW, SUBPLOT.

%   Copyright 1993-2005 The MathWorks, Inc.
%   $Revision: 1.1.8.3 $  $Date: 2005/05/27 14:07:39 $

[x,y,cdata] = parse_inputs(varargin{:});

ax = newplot;
fig = ancestor(ax,'figure');
cm = get(fig,'Colormap');

% Go change all the existing image and texture-mapped surface 
% objects to truecolor.
h = [findobj(fig,'Type','image') ; 
    findobj(fig,'Type','surface','FaceColor','texturemap')];
for k = 1:length(h)
    if (ndims(get(h(k), 'CData')) < 3)
        if (isequal(get(h(k), 'CDataMapping'), 'direct'))
            if (strcmp(get(h(k),'Type'),'image'))
                set(h(k), 'CData', ...
                          iptgate('ind2rgb8',get(h(k),'CData'), cm));
            else
                set(h(k), 'CData', ind2rgb(get(h(k),'CData'), cm));
            end
        else
            clim = get(ancestor(h(k),'axes'), 'CLim');
            data = get(h(k), 'CData');
            if (isa(data,'uint8'))
                data = double(data) / 255;
                clim = clim / 255;
            elseif (isa(data,'uint16'))
                data = double(data) / 65535;
                clim = clim / 65535;
            end
            data = (data - clim(1)) / (clim(2) - clim(1));
            if (strcmp(get(h(k),'Type'),'image'))
                data = im2uint8(data);
                set(h(k), 'CData', cat(3, data, data, data));
            else
                data = min(max(data,0),1);
                set(h(k), 'CData', cat(3, data, data, data));
            end
        end
    end
end

h = image(x, y, cdata);
axis image;

if nargout==1,
    hout = h;
end

%--------------------------------------------------------
% Subfunction PARSE_INPUTS
%--------------------------------------------------------
function [x,y,cdata] = parse_inputs(varargin)

x = [];
y = [];
cdata = [];
msg = '';   % empty string if no error encountered

scaled = 0;
binary = 0;

msg_aEqualsb = 'a cannot equal b in subimage(I,[a b])';
eid_aEqualsb = sprintf('Images:%s:aEqualsB',mfilename);

switch nargin
case 0
    msg = 'Not enough input arguments.';
    eid = sprintf('Images:%s:notEnoughInputs',mfilename);
    error(eid,'%s',msg);    
    
case 1
    % subimage(I)
    % subimage(RGB)
    
    if ((ndims(varargin{1}) == 3) && (size(varargin{1},3) == 3))
        % subimage(RGB)
        cdata = varargin{1};
        
    else
        % subimage(I)
        binary = islogical(varargin{1});
        cdata = cat(3, varargin{1}, varargin{1}, varargin{1});

    end
    
case 2
    % subimage(I,[a b])
    % subimage(I,N)
    % subimage(X,map)
    
    if (numel(varargin{2}) == 1)
        % subimage(I,N)
        binary = islogical(varargin{1});
        cdata = cat(3, varargin{1}, varargin{1}, varargin{1});
        
    elseif (isequal(size(varargin{2}), [1 2]))
        % subimage(I,[a b])
        clim = varargin{2};
        if (clim(1) == clim(2))
            error(eid_aEqualsb,'%s',msg_aEqualsb);
            
        else
            cdata = cat(3, varargin{1}, varargin{1}, varargin{1});
        end
        scaled = 1;
        
    elseif (size(varargin{2},2) == 3)
        % subimage(X,map);
        cdata = iptgate('ind2rgb8',varargin{1},varargin{2});
        
    else
        msg = 'Invalid input arguments.';
        eid = sprintf('Images:%s:invalidInputs',mfilename);
        error(eid,'%s',msg);    
        
    end
        
case 3
    % subimage(x,y,I)
    % subimage(x,y,RGB)
    
    if ((ndims(varargin{3}) == 3) && (size(varargin{3},3) == 3))
        % subimage(x,y,RGB)
        x = varargin{1};
        y = varargin{2};
        cdata = varargin{3};
    
    else
        % subimage(x,y,I)
        x = varargin{1};
        y = varargin{2};
        binary = islogical(varargin{3});
        cdata = cat(3, varargin{3}, varargin{3}, varargin{3});
        
    end
    
case 4
    % subimage(x,y,I,[a b])
    % subimage(x,y,I,N)
    % subimage(x,y,X,map)
    
    if (numel(varargin{4}) == 1)
        % subimage(x,y,I,N)
        x = varargin{1};
        y = varargin{2};
        binary = islogical(varargin{3});
        cdata = cat(3, varargin{3}, varargin{3}, varargin{3});
        
    elseif (isequal(size(varargin{4}), [1 2]))
        % subimage(x,y,I,[a b])
        scaled = 1;
        clim = varargin{4};
        if (clim(1) == clim(2))
            error(eid_aEqualsb,'%s',msg_aEqualsb);
        else            
            x = varargin{1};
            y = varargin{2};
            cdata = cat(3, varargin{3}, varargin{3}, varargin{3});
        end
        
    elseif (size(varargin{4},2) == 3)
        % subimage(x,y,X,map);
        x = varargin{1};
        y = varargin{2};
        cdata = iptgate('ind2rgb8',varargin{3},varargin{4});
        
    else
        msg = 'Invalid input arguments';                
        eid = sprintf('Images:%s:invalidInputs',mfilename);
        error(eid,'%s',msg);            
        
    end
    
otherwise
    msg = 'Too many input arguments';
    eid = sprintf('Images:%s:tooManyInputs',mfilename);
    error(eid,'%s',msg);        
    
end

if (isempty(msg))
    if (scaled)
        if (isa(cdata,'double'))
            cdata = (cdata - clim(1)) / (clim(2) - clim(1));
            cdata = min(max(cdata,0),1);
            
        elseif (isa(cdata,'uint8'))
            cdata = im2double(cdata);
            clim = clim / 255;
            cdata = (cdata - clim(1)) / (clim(2) - clim(1));
            cdata = im2uint8(cdata);
            
        elseif (isa(cdata,'uint16'))
            cdata = im2double(cdata);
            clim = clim / 65535;
            cdata = (cdata - clim(1)) / (clim(2) - clim(1));
            cdata = im2uint8(cdata);
            
        else
            msg = 'Class of input image must be uint8, uint16, or double.';
            eid = sprintf('Images:%s:invalidClass',mfilename);
            error(eid,'%s',msg);    
            
        end
        
    elseif (binary)
        cdata = uint8(cdata);
        cdata(cdata ~= 0) = 255;
    end
    
    if (isempty(x))
        x = [1 size(cdata,2)];
        y = [1 size(cdata,1)];
    end
end

% Regardless of the input type, at this point in the code,
% cdata represents an RGB image; atomatically clip double RGB images 
% to [0 1] range
if isa(cdata, 'double')

   cdata(cdata > 1) = 1;
   cdata(cdata < 0) = 0;
end


function subplot1(M,N,varargin);
%-------------------------------------------------------------------------
% subplot1 function         An mproved subplot function
% Input  : - If more than one input argumenst are given,
%            then the first parameter is the number of rows.
%            If single input argument is given, then this is the
%            subplot-number for which to set focus.
%            This could a scalar or two element vector (I,J).
%          - Number of columns.
%          * variable number of parameters
%            (in pairs: ...,Keywoard, Value,...)
%           - 'Min'    : X, Y lower position of lowest subplot,
%                        default is [0.10 0.10].
%           - 'Max'    : X, Y largest position of highest subplot,
%                        default is [0.95 0.95].
%           - 'Gap'    : X,Y gaps between subplots,
%                        default is [0.01 0.01].
%           - 'XTickL' : x ticks labels option,
%                        'Margin' : plot only XTickLabels in the
%                                   subplot of the lowest  row (default).
%                        'All'    : plot XTickLabels in all subplots.
%                        'None'   : don't plot XTickLabels in subplots.
%           - 'YTickL' : y ticks labels option,
%                        'Margin' : plot only YTickLabels in the
%                                   subplot of the lowest  row (defailt).
%                        'All'    : plot YTickLabels in all subplots.
%                        'None'   : don't plot YTickLabels in subplots.
%           -  'FontS'  : axis font size, default is 10.
%             'XScale' : scale of x axis:
%                        'linear', default.
%                        'log'
%           -  'YScale' : scale of y axis:
%                        'linear', default.
%                        'log'
% Example: subplot1(2,2,'Gap',[0.02 0.02]);
%          subplot1(2,3,'Gap',[0.02 0.02],'XTickL','None','YTickL','All','FontS',16);
% See also : subplot1c.m
% Tested : Matlab 5.3
%     By : Eran O. Ofek           June 2002
%    URL : http://wise-obs.tau.ac.il/~eran/matlab.html
%-------------------------------------------------------------------------
MinDef      = [0.10 0.10];
MaxDef      = [0.95 0.95];
GapDef      = [0.01 0.01];
XTickLDef   = 'Margin';  
YTickLDef   = 'Margin';  
FontSDef    = 10;
XScaleDef   = 'linear';
YScaleDef   = 'linear';

% set default parameters
Min    = MinDef;
Max    = MaxDef;
Gap    = GapDef;
XTickL = XTickLDef;
YTickL = YTickLDef;
FontS  = FontSDef;
XScale = XScaleDef;
YScale = YScaleDef;


MoveFoc = 0;
if (nargin==1),
   %--- move focus to subplot # ---
   MoveFoc = 1;
elseif (nargin==2),
   % do nothing
elseif (nargin>2),
   Narg = length(varargin);
   if (0.5.*Narg==floor(0.5.*Narg)),

      for I=1:2:Narg-1,
         switch varargin{I},
          case 'Min'
 	     Min = varargin{I+1};
          case 'Max'
 	     Max = varargin{I+1};
          case 'Gap'
 	     Gap = varargin{I+1};
          case 'XTickL'
 	     XTickL = varargin{I+1};
          case 'YTickL'
 	     YTickL = varargin{I+1};
          case 'FontS'
 	     FontS = varargin{I+1};
          case 'XScale'
 	     XScale = varargin{I+1};
          case 'YScale'
 	     YScale = varargin{I+1};
          otherwise
	     error('Unknown keyword');
         end
      end
   else
      error('Optional arguments should given as keyword, value');
   end
else
   error('Illegal number of input arguments');
end








switch MoveFoc
 case 1
    %--- move focus to subplot # ---
    H    = get(gcf,'Children');
    Ptot = length(H);
    if (length(M)==1),
       M    = Ptot - M + 1; 
    elseif (length(M)==2),
       %--- check for subplot size ---
       Pos1  = get(H(1),'Position');
       Pos1x = Pos1(1);
       for Icheck=2:1:Ptot,
          PosN  = get(H(Icheck),'Position');
          PosNx = PosN(1);
          if (PosNx==Pos1x),
             NumberOfCol = Icheck - 1;
             break;
          end
       end
       NumberOfRow = Ptot./NumberOfCol;

       Row = M(1);
       Col = M(2);

       M   = (Row-1).*NumberOfCol + Col;
       M    = Ptot - M + 1; 
    else
       error('Unknown option, undefined subplot index');
    end

    set(gcf,'CurrentAxes',H(M));

 case 0
    %--- open subplots ---

    Xmin   = Min(1);
    Ymin   = Min(2);
    Xmax   = Max(1);
    Ymax   = Max(2);
    Xgap   = Gap(1);
    Ygap   = Gap(2);
    
    
    Xsize  = (Xmax - Xmin)./N;
    Ysize  = (Ymax - Ymin)./M;
    
    Xbox   = Xsize - Xgap;
    Ybox   = Ysize - Ygap;
    
    
    Ptot = M.*N;
    
    Hgcf = gcf;
    clf;
    figure(Hgcf);
    for Pi=1:1:Ptot,
       Row = ceil(Pi./N);
       Col = Pi - (Row - 1)*N;

       Xstart = Xmin + Xsize.*(Col - 1);
       Ystart = Ymax - Ysize.*Row;

%       subplot(M,N,Pi);
%       hold on;
       axes('position',[Xstart,Ystart,Xbox,Ybox]);
       %set(gca,'position',[Xstart,Ystart,Xbox,Ybox]);
       set(gca,'FontSize',FontS); 
       box on;
       hold on;

       switch XTickL
        case 'Margin'
           if (Row~=M),
              %--- erase XTickLabel ---
              set(gca,'XTickLabel',[]);
           end
        case 'All'
           % do nothing
        case 'None'
           set(gca,'XTickLabel',[]);
        otherwise
           error('Unknown XTickL option');
       end

       switch YTickL
        case 'Margin'
           if (Col~=1),
              %--- erase YTickLabel ---
              set(gca,'YTickLabel',[]);
           end    
        case 'All'
           % do nothing
        case 'None'
           set(gca,'YTickLabel',[]);
        otherwise
           error('Unknown XTickL option');
       end

       switch XScale
        case 'linear'
           set(gca,'XScale','linear');
        case 'log'
           set(gca,'XScale','log');
        otherwise
  	   error('Unknown XScale option');
       end

       switch YScale
        case 'linear'
           set(gca,'YScale','linear');
        case 'log'
           set(gca,'YScale','log');
        otherwise
  	   error('Unknown YScale option');
       end

    end

 otherwise
    error('Unknown MoveFoc option');
end


function [lambda, xFcn, yFcn, zFcn] = colorMatchFcn(formulary)
%colorMatchFcn  Popular color matching functions.
%
%    [LAMBDA, XFCN, YFCN, ZFCN] = colorMatchFcn(FORMULARY) returns the
%    color matching functions XFCN, YFCN, and ZFCN at each wavelength in
%    the vector LAMBDA.  FORMULARY is a string specifying which set of
%    color matching functions to return.  Supported color matching
%    functions are given below:
%
%    CIE_1931   CIE 1931 2-degree, XYZ
%    1931_FULL  CIE 1931 2-degree, XYZ  (at 1nm resolution)
%    CIE_1964   CIE 1964 10-degree, XYZ
%    1964_FULL  CIE 1964 10-degree, XYZ (at 1nm resolution)
%    Judd       CIE 1931 2-degree, XYZ modified by Judd (1951)
%    Judd_Vos   CIE 1931 2-degree, XYZ modified by Judd (1951) and Vos (1978)
%    Stiles_2   Stiles and Burch 2-degree, RGB (1955)
%    Stiles_10  Stiles and Burch 10-degree, RGB (1959)
%
%    Reference: http://cvrl.ioo.ucl.ac.uk/cmfs.htm 
%
%    See also illuminant.

%    Copyright 1993-2005 The MathWorks, Inc.

switch (lower(formulary))
    case 'judd_vos'
        
        cmf = [380, 2.689900e-003, 2.000000e-004, 1.226000e-002
               385, 5.310500e-003, 3.955600e-004, 2.422200e-002
               390, 1.078100e-002, 8.000000e-004, 4.925000e-002
               395, 2.079200e-002, 1.545700e-003, 9.513500e-002
               400, 3.798100e-002, 2.800000e-003, 1.740900e-001
               405, 6.315700e-002, 4.656200e-003, 2.901300e-001
               410, 9.994100e-002, 7.400000e-003, 4.605300e-001
               415, 1.582400e-001, 1.177900e-002, 7.316600e-001
               420, 2.294800e-001, 1.750000e-002, 1.065800e+000
               425, 2.810800e-001, 2.267800e-002, 1.314600e+000
               430, 3.109500e-001, 2.730000e-002, 1.467200e+000
               435, 3.307200e-001, 3.258400e-002, 1.579600e+000
               440, 3.333600e-001, 3.790000e-002, 1.616600e+000
               445, 3.167200e-001, 4.239100e-002, 1.568200e+000
               450, 2.888200e-001, 4.680000e-002, 1.471700e+000
               455, 2.596900e-001, 5.212200e-002, 1.374000e+000
               460, 2.327600e-001, 6.000000e-002, 1.291700e+000
               465, 2.099900e-001, 7.294200e-002, 1.235600e+000
               470, 1.747600e-001, 9.098000e-002, 1.113800e+000
               475, 1.328700e-001, 1.128400e-001, 9.422000e-001
               480, 9.194400e-002, 1.390200e-001, 7.559600e-001
               485, 5.698500e-002, 1.698700e-001, 5.864000e-001
               490, 3.173100e-002, 2.080200e-001, 4.466900e-001
               495, 1.461300e-002, 2.580800e-001, 3.411600e-001
               500, 4.849100e-003, 3.230000e-001, 2.643700e-001
               505, 2.321500e-003, 4.054000e-001, 2.059400e-001
               510, 9.289900e-003, 5.030000e-001, 1.544500e-001
               515, 2.927800e-002, 6.081100e-001, 1.091800e-001
               520, 6.379100e-002, 7.100000e-001, 7.658500e-002
               525, 1.108100e-001, 7.951000e-001, 5.622700e-002
               530, 1.669200e-001, 8.620000e-001, 4.136600e-002
               535, 2.276800e-001, 9.150500e-001, 2.935300e-002
               540, 2.926900e-001, 9.540000e-001, 2.004200e-002
               545, 3.622500e-001, 9.800400e-001, 1.331200e-002
               550, 4.363500e-001, 9.949500e-001, 8.782300e-003
               555, 5.151300e-001, 1.000100e+000, 5.857300e-003
               560, 5.974800e-001, 9.950000e-001, 4.049300e-003
               565, 6.812100e-001, 9.787500e-001, 2.921700e-003
               570, 7.642500e-001, 9.520000e-001, 2.277100e-003
               575, 8.439400e-001, 9.155800e-001, 1.970600e-003
               580, 9.163500e-001, 8.700000e-001, 1.806600e-003
               585, 9.770300e-001, 8.162300e-001, 1.544900e-003
               590, 1.023000e+000, 7.570000e-001, 1.234800e-003
               595, 1.051300e+000, 6.948300e-001, 1.117700e-003
               600, 1.055000e+000, 6.310000e-001, 9.056400e-004
               605, 1.036200e+000, 5.665400e-001, 6.946700e-004
               610, 9.923900e-001, 5.030000e-001, 4.288500e-004
               615, 9.286100e-001, 4.417200e-001, 3.181700e-004
               620, 8.434600e-001, 3.810000e-001, 2.559800e-004
               625, 7.398300e-001, 3.205200e-001, 1.567900e-004
               630, 6.328900e-001, 2.650000e-001, 9.769400e-005
               635, 5.335100e-001, 2.170200e-001, 6.894400e-005
               640, 4.406200e-001, 1.750000e-001, 5.116500e-005
               645, 3.545300e-001, 1.381200e-001, 3.601600e-005
               650, 2.786200e-001, 1.070000e-001, 2.423800e-005
               655, 2.148500e-001, 8.165200e-002, 1.691500e-005
               660, 1.616100e-001, 6.100000e-002, 1.190600e-005
               665, 1.182000e-001, 4.432700e-002, 8.148900e-006
               670, 8.575300e-002, 3.200000e-002, 5.600600e-006
               675, 6.307700e-002, 2.345400e-002, 3.954400e-006
               680, 4.583400e-002, 1.700000e-002, 2.791200e-006
               685, 3.205700e-002, 1.187200e-002, 1.917600e-006
               690, 2.218700e-002, 8.210000e-003, 1.313500e-006
               695, 1.561200e-002, 5.772300e-003, 9.151900e-007
               700, 1.109800e-002, 4.102000e-003, 6.476700e-007
               705, 7.923300e-003, 2.929100e-003, 4.635200e-007
               710, 5.653100e-003, 2.091000e-003, 3.330400e-007
               715, 4.003900e-003, 1.482200e-003, 2.382300e-007
               720, 2.825300e-003, 1.047000e-003, 1.702600e-007
               725, 1.994700e-003, 7.401500e-004, 1.220700e-007
               730, 1.399400e-003, 5.200000e-004, 8.710700e-008
               735, 9.698000e-004, 3.609300e-004, 6.145500e-008
               740, 6.684700e-004, 2.492000e-004, 4.316200e-008
               745, 4.614100e-004, 1.723100e-004, 3.037900e-008
               750, 3.207300e-004, 1.200000e-004, 2.155400e-008
               755, 2.257300e-004, 8.462000e-005, 1.549300e-008
               760, 1.597300e-004, 6.000000e-005, 1.120400e-008
               765, 1.127500e-004, 4.244600e-005, 8.087300e-009
               770, 7.951300e-005, 3.000000e-005, 5.834000e-009
               775, 5.608700e-005, 2.121000e-005, 4.211000e-009
               780, 3.954100e-005, 1.498900e-005, 3.038300e-009
               785, 2.785200e-005, 1.058400e-005, 2.190700e-009
               790, 1.959700e-005, 7.465600e-006, 1.577800e-009
               795, 1.377000e-005, 5.259200e-006, 1.134800e-009
               800, 9.670000e-006, 3.702800e-006, 8.156500e-010
               805, 6.791800e-006, 2.607600e-006, 5.862600e-010
               810, 4.770600e-006, 1.836500e-006, 4.213800e-010
               815, 3.355000e-006, 1.295000e-006, 3.031900e-010
               820, 2.353400e-006, 9.109200e-007, 2.175300e-010
               825, 1.637700e-006, 6.356400e-007, 1.547600e-010];
           
    case 'judd'
        
        cmf = [370,    0.0008,    0.0001,    0.0046
               380,    0.0045,    0.0004,    0.0224
               390,    0.0201,    0.0015,    0.0925
               400,    0.0611,    0.0045,    0.2799
               410,    0.1267,    0.0093,    0.5835
               420,    0.2285,    0.0175,    1.0622
               430,    0.3081,    0.0273,    1.4526
               440,    0.3312,    0.0379,    1.6064
               450,    0.2888,    0.0468,    1.4717
               460,    0.2323,    0.0600,    1.2880
               470,    0.1745,    0.0910,    1.1133
               480,    0.0920,    0.1390,    0.7552
               490,    0.0318,    0.2080,    0.4461
               500,    0.0048,    0.3230,    0.2644
               510,    0.0093,    0.5030,    0.1541
               520,    0.0636,    0.7100,    0.0763
               530,    0.1668,    0.8620,    0.0412
               540,    0.2926,    0.9540,    0.0200
               550,    0.4364,    0.9950,    0.0088
               560,    0.5970,    0.9950,    0.0039
               570,    0.7642,    0.9520,    0.0020
               580,    0.9159,    0.8700,    0.0016
               590,    1.0225,    0.7570,    0.0011
               600,    1.0544,    0.6310,    0.0007
               610,    0.9922,    0.5030,    0.0003
               620,    0.8432,    0.3810,    0.0002
               630,    0.6327,    0.2650,    0.0001
               640,    0.4404,    0.1750,    0.0000
               650,    0.2787,    0.1070,    0.0000
               660,    0.1619,    0.0610,    0.0000
               670,    0.0858,    0.0320,    0.0000
               680,    0.0459,    0.0170,    0.0000
               690,    0.0222,    0.0082,    0.0000
               700,    0.0113,    0.0041,    0.0000
               710,    0.0057,    0.0021,    0.0000
               720,    0.0028,    0.0011,    0.0000
               730,    0.0015,    0.0005,    0.0000
               740,    0.0005,    0.0002,    0.0000
               750,    0.0003,    0.0001,    0.0000
               760,    0.0002,    0.0001,    0.0000
               770,    0.0001,    0.0000,    0.0000];
           
    case 'cie_1931'
        
        cmf = [360, 0.000129900000, 0.000003917000, 0.000606100000
               365, 0.000232100000, 0.000006965000, 0.001086000000
               370, 0.000414900000, 0.000012390000, 0.001946000000
               375, 0.000741600000, 0.000022020000, 0.003486000000
               380, 0.001368000000, 0.000039000000, 0.006450001000
               385, 0.002236000000, 0.000064000000, 0.010549990000
               390, 0.004243000000, 0.000120000000, 0.020050010000
               395, 0.007650000000, 0.000217000000, 0.036210000000
               400, 0.014310000000, 0.000396000000, 0.067850010000
               405, 0.023190000000, 0.000640000000, 0.110200000000
               410, 0.043510000000, 0.001210000000, 0.207400000000
               415, 0.077630000000, 0.002180000000, 0.371300000000
               420, 0.134380000000, 0.004000000000, 0.645600000000
               425, 0.214770000000, 0.007300000000, 1.039050100000
               430, 0.283900000000, 0.011600000000, 1.385600000000
               435, 0.328500000000, 0.016840000000, 1.622960000000
               440, 0.348280000000, 0.023000000000, 1.747060000000
               445, 0.348060000000, 0.029800000000, 1.782600000000
               450, 0.336200000000, 0.038000000000, 1.772110000000
               455, 0.318700000000, 0.048000000000, 1.744100000000
               460, 0.290800000000, 0.060000000000, 1.669200000000
               465, 0.251100000000, 0.073900000000, 1.528100000000
               470, 0.195360000000, 0.090980000000, 1.287640000000
               475, 0.142100000000, 0.112600000000, 1.041900000000
               480, 0.095640000000, 0.139020000000, 0.812950100000
               485, 0.057950010000, 0.169300000000, 0.616200000000
               490, 0.032010000000, 0.208020000000, 0.465180000000
               495, 0.014700000000, 0.258600000000, 0.353300000000
               500, 0.004900000000, 0.323000000000, 0.272000000000
               505, 0.002400000000, 0.407300000000, 0.212300000000
               510, 0.009300000000, 0.503000000000, 0.158200000000
               515, 0.029100000000, 0.608200000000, 0.111700000000
               520, 0.063270000000, 0.710000000000, 0.078249990000
               525, 0.109600000000, 0.793200000000, 0.057250010000
               530, 0.165500000000, 0.862000000000, 0.042160000000
               535, 0.225749900000, 0.914850100000, 0.029840000000
               540, 0.290400000000, 0.954000000000, 0.020300000000
               545, 0.359700000000, 0.980300000000, 0.013400000000
               550, 0.433449900000, 0.994950100000, 0.008749999000
               555, 0.512050100000, 1.000000000000, 0.005749999000
               560, 0.594500000000, 0.995000000000, 0.003900000000
               565, 0.678400000000, 0.978600000000, 0.002749999000
               570, 0.762100000000, 0.952000000000, 0.002100000000
               575, 0.842500000000, 0.915400000000, 0.001800000000
               580, 0.916300000000, 0.870000000000, 0.001650001000
               585, 0.978600000000, 0.816300000000, 0.001400000000
               590, 1.026300000000, 0.757000000000, 0.001100000000
               595, 1.056700000000, 0.694900000000, 0.001000000000
               600, 1.062200000000, 0.631000000000, 0.000800000000
               605, 1.045600000000, 0.566800000000, 0.000600000000
               610, 1.002600000000, 0.503000000000, 0.000340000000
               615, 0.938400000000, 0.441200000000, 0.000240000000
               620, 0.854449900000, 0.381000000000, 0.000190000000
               625, 0.751400000000, 0.321000000000, 0.000100000000
               630, 0.642400000000, 0.265000000000, 0.000049999990
               635, 0.541900000000, 0.217000000000, 0.000030000000
               640, 0.447900000000, 0.175000000000, 0.000020000000
               645, 0.360800000000, 0.138200000000, 0.000010000000
               650, 0.283500000000, 0.107000000000, 0.000000000000
               655, 0.218700000000, 0.081600000000, 0.000000000000
               660, 0.164900000000, 0.061000000000, 0.000000000000
               665, 0.121200000000, 0.044580000000, 0.000000000000
               670, 0.087400000000, 0.032000000000, 0.000000000000
               675, 0.063600000000, 0.023200000000, 0.000000000000
               680, 0.046770000000, 0.017000000000, 0.000000000000
               685, 0.032900000000, 0.011920000000, 0.000000000000
               690, 0.022700000000, 0.008210000000, 0.000000000000
               695, 0.015840000000, 0.005723000000, 0.000000000000
               700, 0.011359160000, 0.004102000000, 0.000000000000
               705, 0.008110916000, 0.002929000000, 0.000000000000
               710, 0.005790346000, 0.002091000000, 0.000000000000
               715, 0.004109457000, 0.001484000000, 0.000000000000
               720, 0.002899327000, 0.001047000000, 0.000000000000
               725, 0.002049190000, 0.000740000000, 0.000000000000
               730, 0.001439971000, 0.000520000000, 0.000000000000
               735, 0.000999949300, 0.000361100000, 0.000000000000
               740, 0.000690078600, 0.000249200000, 0.000000000000
               745, 0.000476021300, 0.000171900000, 0.000000000000
               750, 0.000332301100, 0.000120000000, 0.000000000000
               755, 0.000234826100, 0.000084800000, 0.000000000000
               760, 0.000166150500, 0.000060000000, 0.000000000000
               765, 0.000117413000, 0.000042400000, 0.000000000000
               770, 0.000083075270, 0.000030000000, 0.000000000000
               775, 0.000058706520, 0.000021200000, 0.000000000000
               780, 0.000041509940, 0.000014990000, 0.000000000000
               785, 0.000029353260, 0.000010600000, 0.000000000000
               790, 0.000020673830, 0.000007465700, 0.000000000000
               795, 0.000014559770, 0.000005257800, 0.000000000000
               800, 0.000010253980, 0.000003702900, 0.000000000000
               805, 0.000007221456, 0.000002607800, 0.000000000000
               810, 0.000005085868, 0.000001836600, 0.000000000000
               815, 0.000003581652, 0.000001293400, 0.000000000000
               820, 0.000002522525, 0.000000910930, 0.000000000000
               825, 0.000001776509, 0.000000641530, 0.000000000000
               830, 0.000001251141, 0.000000451810, 0.000000000000];

    case 'stiles_2'
        
        cmf = [390,  1.83970e-003, -4.53930e-004,  1.21520e-002
               395,  4.61530e-003, -1.04640e-003,  3.11100e-002
               400,  9.62640e-003, -2.16890e-003,  6.23710e-002
               405,  1.89790e-002, -4.43040e-003,  1.31610e-001
               410,  3.08030e-002, -7.20480e-003,  2.27500e-001
               415,  4.24590e-002, -1.25790e-002,  3.58970e-001
               420,  5.16620e-002, -1.66510e-002,  5.23960e-001
               425,  5.28370e-002, -2.12400e-002,  6.85860e-001
               430,  4.42870e-002, -1.99360e-002,  7.96040e-001
               435,  3.22200e-002, -1.60970e-002,  8.94590e-001
               440,  1.47630e-002, -7.34570e-003,  9.63950e-001
               445, -2.33920e-003,  1.36900e-003,  9.98140e-001
               450, -2.91300e-002,  1.96100e-002,  9.18750e-001
               455, -6.06770e-002,  4.34640e-002,  8.24870e-001
               460, -9.62240e-002,  7.09540e-002,  7.85540e-001
               465, -1.37590e-001,  1.10220e-001,  6.67230e-001
               470, -1.74860e-001,  1.50880e-001,  6.10980e-001
               475, -2.12600e-001,  1.97940e-001,  4.88290e-001
               480, -2.37800e-001,  2.40420e-001,  3.61950e-001
               485, -2.56740e-001,  2.79930e-001,  2.66340e-001
               490, -2.77270e-001,  3.33530e-001,  1.95930e-001
               495, -2.91250e-001,  4.05210e-001,  1.47300e-001
               500, -2.95000e-001,  4.90600e-001,  1.07490e-001
               505, -2.97060e-001,  5.96730e-001,  7.67140e-002
               510, -2.67590e-001,  7.01840e-001,  5.02480e-002
               515, -2.17250e-001,  8.08520e-001,  2.87810e-002
               520, -1.47680e-001,  9.10760e-001,  1.33090e-002
               525, -3.51840e-002,  9.84820e-001,  2.11700e-003
               530,  1.06140e-001,  1.03390e+000, -4.15740e-003
               535,  2.59810e-001,  1.05380e+000, -8.30320e-003
               540,  4.19760e-001,  1.05120e+000, -1.21910e-002
               545,  5.92590e-001,  1.04980e+000, -1.40390e-002
               550,  7.90040e-001,  1.03680e+000, -1.46810e-002
               555,  1.00780e+000,  9.98260e-001, -1.49470e-002
               560,  1.22830e+000,  9.37830e-001, -1.46130e-002
               565,  1.47270e+000,  8.80390e-001, -1.37820e-002
               570,  1.74760e+000,  8.28350e-001, -1.26500e-002
               575,  2.02140e+000,  7.46860e-001, -1.13560e-002
               580,  2.27240e+000,  6.49300e-001, -9.93170e-003
               585,  2.48960e+000,  5.63170e-001, -8.41480e-003
               590,  2.67250e+000,  4.76750e-001, -7.02100e-003
               595,  2.80930e+000,  3.84840e-001, -5.74370e-003
               600,  2.87170e+000,  3.00690e-001, -4.27430e-003
               605,  2.85250e+000,  2.28530e-001, -2.91320e-003
               610,  2.76010e+000,  1.65750e-001, -2.26930e-003
               615,  2.59890e+000,  1.13730e-001, -1.99660e-003
               620,  2.37430e+000,  7.46820e-002, -1.50690e-003
               625,  2.10540e+000,  4.65040e-002, -9.38220e-004
               630,  1.81450e+000,  2.63330e-002, -5.53160e-004
               635,  1.52470e+000,  1.27240e-002, -3.16680e-004
               640,  1.25430e+000,  4.50330e-003, -1.43190e-004
               645,  1.00760e+000,  9.66110e-005, -4.08310e-006
               650,  7.86420e-001, -1.96450e-003,  1.10810e-004
               655,  5.96590e-001, -2.63270e-003,  1.91750e-004
               660,  4.43200e-001, -2.62620e-003,  2.26560e-004
               665,  3.24100e-001, -2.30270e-003,  2.15200e-004
               670,  2.34550e-001, -1.87000e-003,  1.63610e-004
               675,  1.68840e-001, -1.44240e-003,  9.71640e-005
               680,  1.20860e-001, -1.07550e-003,  5.10330e-005
               685,  8.58110e-002, -7.90040e-004,  3.52710e-005
               690,  6.02600e-002, -5.67650e-004,  3.12110e-005
               695,  4.14800e-002, -3.92740e-004,  2.45080e-005
               700,  2.81140e-002, -2.62310e-004,  1.65210e-005
               705,  1.91170e-002, -1.75120e-004,  1.11240e-005
               710,  1.33050e-002, -1.21400e-004,  8.69650e-006
               715,  9.40920e-003, -8.57600e-005,  7.43510e-006
               720,  6.51770e-003, -5.76770e-005,  6.10570e-006
               725,  4.53770e-003, -3.90030e-005,  5.02770e-006
               730,  3.17420e-003, -2.65110e-005,  4.12510e-006];

    case 'stiles_10'
        
        cmf = [390,   1.5000E-03,  -4.0000E-04,   6.2000E-03
               395,   3.8000E-03,  -1.0000E-03,   1.6100E-02
               400,   8.9000E-03,  -2.5000E-03,   4.0000E-02
               405,   1.8800E-02,  -5.9000E-03,   9.0600E-02
               410,   3.5000E-02,  -1.1900E-02,   1.8020E-01
               415,   5.3100E-02,  -2.0100E-02,   3.0880E-01
               420,   7.0200E-02,  -2.8900E-02,   4.6700E-01
               425,   7.6300E-02,  -3.3800E-02,   6.1520E-01
               430,   7.4500E-02,  -3.4900E-02,   7.6380E-01
               435,   5.6100E-02,  -2.7600E-02,   8.7780E-01
               440,   3.2300E-02,  -1.6900E-02,   9.7550E-01
               445,  -4.4000E-03,   2.4000E-03,   1.0019E+00
               450,  -4.7800E-02,   2.8300E-02,   9.9960E-01
               455,  -9.7000E-02,   6.3600E-02,   9.1390E-01
               460,  -1.5860E-01,   1.0820E-01,   8.2970E-01
               465,  -2.2350E-01,   1.6170E-01,   7.4170E-01
               470,  -2.8480E-01,   2.2010E-01,   6.1340E-01
               475,  -3.3460E-01,   2.7960E-01,   4.7200E-01
               480,  -3.7760E-01,   3.4280E-01,   3.4950E-01
               485,  -4.1360E-01,   4.0860E-01,   2.5640E-01
               490,  -4.3170E-01,   4.7160E-01,   1.8190E-01
               495,  -4.4520E-01,   5.4910E-01,   1.3070E-01
               500,  -4.3500E-01,   6.2600E-01,   9.1000E-02
               505,  -4.1400E-01,   7.0970E-01,   5.8000E-02
               510,  -3.6730E-01,   7.9350E-01,   3.5700E-02
               515,  -2.8450E-01,   8.7150E-01,   2.0000E-02
               520,  -1.8550E-01,   9.4770E-01,   9.5000E-03
               525,  -4.3500E-02,   9.9450E-01,   7.0000E-04
               530,   1.2700E-01,   1.0203E+00,  -4.3000E-03
               535,   3.1290E-01,   1.0375E+00,  -6.4000E-03
               540,   5.3620E-01,   1.0517E+00,  -8.2000E-03
               545,   7.7220E-01,   1.0390E+00,  -9.4000E-03
               550,   1.0059E+00,   1.0029E+00,  -9.7000E-03
               555,   1.2710E+00,   9.6980E-01,  -9.7000E-03
               560,   1.5574E+00,   9.1620E-01,  -9.3000E-03
               565,   1.8465E+00,   8.5710E-01,  -8.7000E-03
               570,   2.1511E+00,   7.8230E-01,  -8.0000E-03
               575,   2.4250E+00,   6.9530E-01,  -7.3000E-03
               580,   2.6574E+00,   5.9660E-01,  -6.3000E-03
               585,   2.9151E+00,   5.0630E-01,  -5.3700E-03
               590,   3.0779E+00,   4.2030E-01,  -4.4500E-03
               595,   3.1613E+00,   3.3600E-01,  -3.5700E-03
               600,   3.1673E+00,   2.5910E-01,  -2.7700E-03
               605,   3.1048E+00,   1.9170E-01,  -2.0800E-03
               610,   2.9462E+00,   1.3670E-01,  -1.5000E-03
               615,   2.7194E+00,   9.3800E-02,  -1.0300E-03
               620,   2.4526E+00,   6.1100E-02,  -6.8000E-04
               625,   2.1700E+00,   3.7100E-02,  -4.4200E-04
               630,   1.8358E+00,   2.1500E-02,  -2.7200E-04
               635,   1.5179E+00,   1.1200E-02,  -1.4100E-04
               640,   1.2428E+00,   4.4000E-03,  -5.4900E-05
               645,   1.0070E+00,   7.8000E-05,  -2.2000E-06
               650,   7.8270E-01,  -1.3680E-03,   2.3700E-05
               655,   5.9340E-01,  -1.9880E-03,   2.8600E-05
               660,   4.4420E-01,  -2.1680E-03,   2.6100E-05
               665,   3.2830E-01,  -2.0060E-03,   2.2500E-05
               670,   2.3940E-01,  -1.6420E-03,   1.8200E-05
               675,   1.7220E-01,  -1.2720E-03,   1.3900E-05
               680,   1.2210E-01,  -9.4700E-04,   1.0300E-05
               685,   8.5300E-02,  -6.8300E-04,   7.3800E-06
               690,   5.8600E-02,  -4.7800E-04,   5.2200E-06
               695,   4.0800E-02,  -3.3700E-04,   3.6700E-06
               700,   2.8400E-02,  -2.3500E-04,   2.5600E-06
               705,   1.9700E-02,  -1.6300E-04,   1.7600E-06
               710,   1.3500E-02,  -1.1100E-04,   1.2000E-06
               715,   9.2400E-03,  -7.4800E-05,   8.1700E-07
               720,   6.3800E-03,  -5.0800E-05,   5.5500E-07
               725,   4.4100E-03,  -3.4400E-05,   3.7500E-07
               730,   3.0700E-03,  -2.3400E-05,   2.5400E-07
               735,   2.1400E-03,  -1.5900E-05,   1.7100E-07
               740,   1.4900E-03,  -1.0700E-05,   1.1600E-07
               745,   1.0500E-03,  -7.2300E-06,   7.8500E-08
               750,   7.3900E-04,  -4.8700E-06,   5.3100E-08
               755,   5.2300E-04,  -3.2900E-06,   3.6000E-08
               760,   3.7200E-04,  -2.2200E-06,   2.4400E-08
               765,   2.6500E-04,  -1.5000E-06,   1.6500E-08
               770,   1.9000E-04,  -1.0200E-06,   1.1200E-08
               775,   1.3600E-04,  -6.8800E-07,   7.5300E-09
               780,   9.8400E-05,  -4.6500E-07,   5.0700E-09
               785,   7.1300E-05,  -3.1200E-07,   3.4000E-09
               790,   5.1800E-05,  -2.0800E-07,   2.2700E-09
               795,   3.7700E-05,  -1.3700E-07,   1.5000E-09
               800,   2.7600E-05,  -8.8000E-08,   9.8600E-10
               805,   2.0300E-05,  -5.5300E-08,   6.3900E-10
               810,   1.4900E-05,  -3.3600E-08,   4.0700E-10
               815,   1.1000E-05,  -1.9600E-08,   2.5300E-10
               820,   8.1800E-06,  -1.0900E-08,   1.5200E-10
               825,   6.0900E-06,  -5.7000E-09,   8.6400E-11
               830,   4.5500E-06,  -2.7700E-09,   4.4200E-11];

    case 'cie_1964'
        
        cmf = [360, 0.000000122200, 0.000000013398, 0.000000535027
               365, 0.000000919270, 0.000000100650, 0.000004028300
               370, 0.000005958600, 0.000000651100, 0.000026143700
               375, 0.000033266000, 0.000003625000, 0.000146220000
               380, 0.000159952000, 0.000017364000, 0.000704776000
               385, 0.000662440000, 0.000071560000, 0.002927800000
               390, 0.002361600000, 0.000253400000, 0.010482200000
               395, 0.007242300000, 0.000768500000, 0.032344000000
               400, 0.019109700000, 0.002004400000, 0.086010900000
               405, 0.043400000000, 0.004509000000, 0.197120000000
               410, 0.084736000000, 0.008756000000, 0.389366000000
               415, 0.140638000000, 0.014456000000, 0.656760000000
               420, 0.204492000000, 0.021391000000, 0.972542000000
               425, 0.264737000000, 0.029497000000, 1.282500000000
               430, 0.314679000000, 0.038676000000, 1.553480000000
               435, 0.357719000000, 0.049602000000, 1.798500000000
               440, 0.383734000000, 0.062077000000, 1.967280000000
               445, 0.386726000000, 0.074704000000, 2.027300000000
               450, 0.370702000000, 0.089456000000, 1.994800000000
               455, 0.342957000000, 0.106256000000, 1.900700000000
               460, 0.302273000000, 0.128201000000, 1.745370000000
               465, 0.254085000000, 0.152761000000, 1.554900000000
               470, 0.195618000000, 0.185190000000, 1.317560000000
               475, 0.132349000000, 0.219940000000, 1.030200000000
               480, 0.080507000000, 0.253589000000, 0.772125000000
               485, 0.041072000000, 0.297665000000, 0.570060000000
               490, 0.016172000000, 0.339133000000, 0.415254000000
               495, 0.005132000000, 0.395379000000, 0.302356000000
               500, 0.003816000000, 0.460777000000, 0.218502000000
               505, 0.015444000000, 0.531360000000, 0.159249000000
               510, 0.037465000000, 0.606741000000, 0.112044000000
               515, 0.071358000000, 0.685660000000, 0.082248000000
               520, 0.117749000000, 0.761757000000, 0.060709000000
               525, 0.172953000000, 0.823330000000, 0.043050000000
               530, 0.236491000000, 0.875211000000, 0.030451000000
               535, 0.304213000000, 0.923810000000, 0.020584000000
               540, 0.376772000000, 0.961988000000, 0.013676000000
               545, 0.451584000000, 0.982200000000, 0.007918000000
               550, 0.529826000000, 0.991761000000, 0.003988000000
               555, 0.616053000000, 0.999110000000, 0.001091000000
               560, 0.705224000000, 0.997340000000, 0
               565, 0.793832000000, 0.982380000000, 0
               570, 0.878655000000, 0.955552000000, 0
               575, 0.951162000000, 0.915175000000, 0
               580, 1.014160000000, 0.868934000000, 0
               585, 1.074300000000, 0.825623000000, 0
               590, 1.118520000000, 0.777405000000, 0
               595, 1.134300000000, 0.720353000000, 0
               600, 1.123990000000, 0.658341000000, 0
               605, 1.089100000000, 0.593878000000, 0
               610, 1.030480000000, 0.527963000000, 0
               615, 0.950740000000, 0.461834000000, 0
               620, 0.856297000000, 0.398057000000, 0
               625, 0.754930000000, 0.339554000000, 0
               630, 0.647467000000, 0.283493000000, 0
               635, 0.535110000000, 0.228254000000, 0
               640, 0.431567000000, 0.179828000000, 0
               645, 0.343690000000, 0.140211000000, 0
               650, 0.268329000000, 0.107633000000, 0
               655, 0.204300000000, 0.081187000000, 0
               660, 0.152568000000, 0.060281000000, 0
               665, 0.112210000000, 0.044096000000, 0
               670, 0.081260600000, 0.031800400000, 0
               675, 0.057930000000, 0.022601700000, 0
               680, 0.040850800000, 0.015905100000, 0
               685, 0.028623000000, 0.011130300000, 0
               690, 0.019941300000, 0.007748800000, 0
               695, 0.013842000000, 0.005375100000, 0
               700, 0.009576880000, 0.003717740000, 0
               705, 0.006605200000, 0.002564560000, 0
               710, 0.004552630000, 0.001768470000, 0
               715, 0.003144700000, 0.001222390000, 0
               720, 0.002174960000, 0.000846190000, 0
               725, 0.001505700000, 0.000586440000, 0
               730, 0.001044760000, 0.000407410000, 0
               735, 0.000727450000, 0.000284041000, 0
               740, 0.000508258000, 0.000198730000, 0
               745, 0.000356380000, 0.000139550000, 0
               750, 0.000250969000, 0.000098428000, 0
               755, 0.000177730000, 0.000069819000, 0
               760, 0.000126390000, 0.000049737000, 0
               765, 0.000090151000, 0.000035540500, 0
               770, 0.000064525800, 0.000025486000, 0
               775, 0.000046339000, 0.000018338400, 0
               780, 0.000033411700, 0.000013249000, 0
               785, 0.000024209000, 0.000009619600, 0
               790, 0.000017611500, 0.000007012800, 0
               795, 0.000012855000, 0.000005129800, 0
               800, 0.000009413630, 0.000003764730, 0
               805, 0.000006913000, 0.000002770810, 0
               810, 0.000005093470, 0.000002046130, 0
               815, 0.000003767100, 0.000001516770, 0
               820, 0.000002795310, 0.000001128090, 0
               825, 0.000002082000, 0.000000842160, 0
               830, 0.000001553140, 0.000000629700, 0];

    case '1931_full'
        
        cmf = [360,  0.000129900000,  0.000003917000,  0.000606100000
               361,  0.000145847000,  0.000004393581,  0.000680879200
               362,  0.000163802100,  0.000004929604,  0.000765145600
               363,  0.000184003700,  0.000005532136,  0.000860012400
               364,  0.000206690200,  0.000006208245,  0.000966592800
               365,  0.000232100000,  0.000006965000,  0.001086000000
               366,  0.000260728000,  0.000007813219,  0.001220586000
               367,  0.000293075000,  0.000008767336,  0.001372729000
               368,  0.000329388000,  0.000009839844,  0.001543579000
               369,  0.000369914000,  0.000011043230,  0.001734286000
               370,  0.000414900000,  0.000012390000,  0.001946000000
               371,  0.000464158700,  0.000013886410,  0.002177777000
               372,  0.000518986000,  0.000015557280,  0.002435809000
               373,  0.000581854000,  0.000017442960,  0.002731953000
               374,  0.000655234700,  0.000019583750,  0.003078064000
               375,  0.000741600000,  0.000022020000,  0.003486000000
               376,  0.000845029600,  0.000024839650,  0.003975227000
               377,  0.000964526800,  0.000028041260,  0.004540880000
               378,  0.001094949000,  0.000031531040,  0.005158320000
               379,  0.001231154000,  0.000035215210,  0.005802907000
               380,  0.001368000000,  0.000039000000,  0.006450001000
               381,  0.001502050000,  0.000042826400,  0.007083216000
               382,  0.001642328000,  0.000046914600,  0.007745488000
               383,  0.001802382000,  0.000051589600,  0.008501152000
               384,  0.001995757000,  0.000057176400,  0.009414544000
               385,  0.002236000000,  0.000064000000,  0.010549990000
               386,  0.002535385000,  0.000072344210,  0.011965800000
               387,  0.002892603000,  0.000082212240,  0.013655870000
               388,  0.003300829000,  0.000093508160,  0.015588050000
               389,  0.003753236000,  0.000106136100,  0.017730150000
               390,  0.004243000000,  0.000120000000,  0.020050010000
               391,  0.004762389000,  0.000134984000,  0.022511360000
               392,  0.005330048000,  0.000151492000,  0.025202880000
               393,  0.005978712000,  0.000170208000,  0.028279720000
               394,  0.006741117000,  0.000191816000,  0.031897040000
               395,  0.007650000000,  0.000217000000,  0.036210000000
               396,  0.008751373000,  0.000246906700,  0.041437710000
               397,  0.010028880000,  0.000281240000,  0.047503720000
               398,  0.011421700000,  0.000318520000,  0.054119880000
               399,  0.012869010000,  0.000357266700,  0.060998030000
               400,  0.014310000000,  0.000396000000,  0.067850010000
               401,  0.015704430000,  0.000433714700,  0.074486320000
               402,  0.017147440000,  0.000473024000,  0.081361560000
               403,  0.018781220000,  0.000517876000,  0.089153640000
               404,  0.020748010000,  0.000572218700,  0.098540480000
               405,  0.023190000000,  0.000640000000,  0.110200000000
               406,  0.026207360000,  0.000724560000,  0.124613300000
               407,  0.029782480000,  0.000825500000,  0.141701700000
               408,  0.033880920000,  0.000941160000,  0.161303500000
               409,  0.038468240000,  0.001069880000,  0.183256800000
               410,  0.043510000000,  0.001210000000,  0.207400000000
               411,  0.048995600000,  0.001362091000,  0.233692100000
               412,  0.055022600000,  0.001530752000,  0.262611400000
               413,  0.061718800000,  0.001720368000,  0.294774600000
               414,  0.069212000000,  0.001935323000,  0.330798500000
               415,  0.077630000000,  0.002180000000,  0.371300000000
               416,  0.086958110000,  0.002454800000,  0.416209100000
               417,  0.097176720000,  0.002764000000,  0.465464200000
               418,  0.108406300000,  0.003117800000,  0.519694800000
               419,  0.120767200000,  0.003526400000,  0.579530300000
               420,  0.134380000000,  0.004000000000,  0.645600000000
               421,  0.149358200000,  0.004546240000,  0.718483800000
               422,  0.165395700000,  0.005159320000,  0.796713300000
               423,  0.181983100000,  0.005829280000,  0.877845900000
               424,  0.198611000000,  0.006546160000,  0.959439000000
               425,  0.214770000000,  0.007300000000,  1.039050100000
               426,  0.230186800000,  0.008086507000,  1.115367300000
               427,  0.244879700000,  0.008908720000,  1.188497100000
               428,  0.258777300000,  0.009767680000,  1.258123300000
               429,  0.271807900000,  0.010664430000,  1.323929600000
               430,  0.283900000000,  0.011600000000,  1.385600000000
               431,  0.294943800000,  0.012573170000,  1.442635200000
               432,  0.304896500000,  0.013582720000,  1.494803500000
               433,  0.313787300000,  0.014629680000,  1.542190300000
               434,  0.321645400000,  0.015715090000,  1.584880700000
               435,  0.328500000000,  0.016840000000,  1.622960000000
               436,  0.334351300000,  0.018007360000,  1.656404800000
               437,  0.339210100000,  0.019214480000,  1.685295900000
               438,  0.343121300000,  0.020453920000,  1.709874500000
               439,  0.346129600000,  0.021718240000,  1.730382100000
               440,  0.348280000000,  0.023000000000,  1.747060000000
               441,  0.349599900000,  0.024294610000,  1.760044600000
               442,  0.350147400000,  0.025610240000,  1.769623300000
               443,  0.350013000000,  0.026958570000,  1.776263700000
               444,  0.349287000000,  0.028351250000,  1.780433400000
               445,  0.348060000000,  0.029800000000,  1.782600000000
               446,  0.346373300000,  0.031310830000,  1.782968200000
               447,  0.344262400000,  0.032883680000,  1.781699800000
               448,  0.341808800000,  0.034521120000,  1.779198200000
               449,  0.339094100000,  0.036225710000,  1.775867100000
               450,  0.336200000000,  0.038000000000,  1.772110000000
               451,  0.333197700000,  0.039846670000,  1.768258900000
               452,  0.330041100000,  0.041768000000,  1.764039000000
               453,  0.326635700000,  0.043766000000,  1.758943800000
               454,  0.322886800000,  0.045842670000,  1.752466300000
               455,  0.318700000000,  0.048000000000,  1.744100000000
               456,  0.314025100000,  0.050243680000,  1.733559500000
               457,  0.308884000000,  0.052573040000,  1.720858100000
               458,  0.303290400000,  0.054980560000,  1.705936900000
               459,  0.297257900000,  0.057458720000,  1.688737200000
               460,  0.290800000000,  0.060000000000,  1.669200000000
               461,  0.283970100000,  0.062601970000,  1.647528700000
               462,  0.276721400000,  0.065277520000,  1.623412700000
               463,  0.268917800000,  0.068042080000,  1.596022300000
               464,  0.260422700000,  0.070911090000,  1.564528000000
               465,  0.251100000000,  0.073900000000,  1.528100000000
               466,  0.240847500000,  0.077016000000,  1.486111400000
               467,  0.229851200000,  0.080266400000,  1.439521500000
               468,  0.218407200000,  0.083666800000,  1.389879900000
               469,  0.206811500000,  0.087232800000,  1.338736200000
               470,  0.195360000000,  0.090980000000,  1.287640000000
               471,  0.184213600000,  0.094917550000,  1.237422300000
               472,  0.173327300000,  0.099045840000,  1.187824300000
               473,  0.162688100000,  0.103367400000,  1.138761100000
               474,  0.152283300000,  0.107884600000,  1.090148000000
               475,  0.142100000000,  0.112600000000,  1.041900000000
               476,  0.132178600000,  0.117532000000,  0.994197600000
               477,  0.122569600000,  0.122674400000,  0.947347300000
               478,  0.113275200000,  0.127992800000,  0.901453100000
               479,  0.104297900000,  0.133452800000,  0.856619300000
               480,  0.095640000000,  0.139020000000,  0.812950100000
               481,  0.087299550000,  0.144676400000,  0.770517300000
               482,  0.079308040000,  0.150469300000,  0.729444800000
               483,  0.071717760000,  0.156461900000,  0.689913600000
               484,  0.064580990000,  0.162717700000,  0.652104900000
               485,  0.057950010000,  0.169300000000,  0.616200000000
               486,  0.051862110000,  0.176243100000,  0.582328600000
               487,  0.046281520000,  0.183558100000,  0.550416200000
               488,  0.041150880000,  0.191273500000,  0.520337600000
               489,  0.036412830000,  0.199418000000,  0.491967300000
               490,  0.032010000000,  0.208020000000,  0.465180000000
               491,  0.027917200000,  0.217119900000,  0.439924600000
               492,  0.024144400000,  0.226734500000,  0.416183600000
               493,  0.020687000000,  0.236857100000,  0.393882200000
               494,  0.017540400000,  0.247481200000,  0.372945900000
               495,  0.014700000000,  0.258600000000,  0.353300000000
               496,  0.012161790000,  0.270184900000,  0.334857800000
               497,  0.009919960000,  0.282293900000,  0.317552100000
               498,  0.007967240000,  0.295050500000,  0.301337500000
               499,  0.006296346000,  0.308578000000,  0.286168600000
               500,  0.004900000000,  0.323000000000,  0.272000000000
               501,  0.003777173000,  0.338402100000,  0.258817100000
               502,  0.002945320000,  0.354685800000,  0.246483800000
               503,  0.002424880000,  0.371698600000,  0.234771800000
               504,  0.002236293000,  0.389287500000,  0.223453300000
               505,  0.002400000000,  0.407300000000,  0.212300000000
               506,  0.002925520000,  0.425629900000,  0.201169200000
               507,  0.003836560000,  0.444309600000,  0.190119600000
               508,  0.005174840000,  0.463394400000,  0.179225400000
               509,  0.006982080000,  0.482939500000,  0.168560800000
               510,  0.009300000000,  0.503000000000,  0.158200000000
               511,  0.012149490000,  0.523569300000,  0.148138300000
               512,  0.015535880000,  0.544512000000,  0.138375800000
               513,  0.019477520000,  0.565690000000,  0.128994200000
               514,  0.023992770000,  0.586965300000,  0.120075100000
               515,  0.029100000000,  0.608200000000,  0.111700000000
               516,  0.034814850000,  0.629345600000,  0.103904800000
               517,  0.041120160000,  0.650306800000,  0.096667480000
               518,  0.047985040000,  0.670875200000,  0.089982720000
               519,  0.055378610000,  0.690842400000,  0.083845310000
               520,  0.063270000000,  0.710000000000,  0.078249990000
               521,  0.071635010000,  0.728185200000,  0.073208990000
               522,  0.080462240000,  0.745463600000,  0.068678160000
               523,  0.089739960000,  0.761969400000,  0.064567840000
               524,  0.099456450000,  0.777836800000,  0.060788350000
               525,  0.109600000000,  0.793200000000,  0.057250010000
               526,  0.120167400000,  0.808110400000,  0.053904350000
               527,  0.131114500000,  0.822496200000,  0.050746640000
               528,  0.142367900000,  0.836306800000,  0.047752760000
               529,  0.153854200000,  0.849491600000,  0.044898590000
               530,  0.165500000000,  0.862000000000,  0.042160000000
               531,  0.177257100000,  0.873810800000,  0.039507280000
               532,  0.189140000000,  0.884962400000,  0.036935640000
               533,  0.201169400000,  0.895493600000,  0.034458360000
               534,  0.213365800000,  0.905443200000,  0.032088720000
               535,  0.225749900000,  0.914850100000,  0.029840000000
               536,  0.238320900000,  0.923734800000,  0.027711810000
               537,  0.251066800000,  0.932092400000,  0.025694440000
               538,  0.263992200000,  0.939922600000,  0.023787160000
               539,  0.277101700000,  0.947225200000,  0.021989250000
               540,  0.290400000000,  0.954000000000,  0.020300000000
               541,  0.303891200000,  0.960256100000,  0.018718050000
               542,  0.317572600000,  0.966007400000,  0.017240360000
               543,  0.331438400000,  0.971260600000,  0.015863640000
               544,  0.345482800000,  0.976022500000,  0.014584610000
               545,  0.359700000000,  0.980300000000,  0.013400000000
               546,  0.374083900000,  0.984092400000,  0.012307230000
               547,  0.388639600000,  0.987418200000,  0.011301880000
               548,  0.403378400000,  0.990312800000,  0.010377920000
               549,  0.418311500000,  0.992811600000,  0.009529306000
               550,  0.433449900000,  0.994950100000,  0.008749999000
               551,  0.448795300000,  0.996710800000,  0.008035200000
               552,  0.464336000000,  0.998098300000,  0.007381600000
               553,  0.480064000000,  0.999112000000,  0.006785400000
               554,  0.495971300000,  0.999748200000,  0.006242800000
               555,  0.512050100000,  1.000000000000,  0.005749999000
               556,  0.528295900000,  0.999856700000,  0.005303600000
               557,  0.544691600000,  0.999304600000,  0.004899800000
               558,  0.561209400000,  0.998325500000,  0.004534200000
               559,  0.577821500000,  0.996898700000,  0.004202400000
               560,  0.594500000000,  0.995000000000,  0.003900000000
               561,  0.611220900000,  0.992600500000,  0.003623200000
               562,  0.627975800000,  0.989742600000,  0.003370600000
               563,  0.644760200000,  0.986444400000,  0.003141400000
               564,  0.661569700000,  0.982724100000,  0.002934800000
               565,  0.678400000000,  0.978600000000,  0.002749999000
               566,  0.695239200000,  0.974083700000,  0.002585200000
               567,  0.712058600000,  0.969171200000,  0.002438600000
               568,  0.728828400000,  0.963856800000,  0.002309400000
               569,  0.745518800000,  0.958134900000,  0.002196800000
               570,  0.762100000000,  0.952000000000,  0.002100000000
               571,  0.778543200000,  0.945450400000,  0.002017733000
               572,  0.794825600000,  0.938499200000,  0.001948200000
               573,  0.810926400000,  0.931162800000,  0.001889800000
               574,  0.826824800000,  0.923457600000,  0.001840933000
               575,  0.842500000000,  0.915400000000,  0.001800000000
               576,  0.857932500000,  0.907006400000,  0.001766267000
               577,  0.873081600000,  0.898277200000,  0.001737800000
               578,  0.887894400000,  0.889204800000,  0.001711200000
               579,  0.902318100000,  0.879781600000,  0.001683067000
               580,  0.916300000000,  0.870000000000,  0.001650001000
               581,  0.929799500000,  0.859861300000,  0.001610133000
               582,  0.942798400000,  0.849392000000,  0.001564400000
               583,  0.955277600000,  0.838622000000,  0.001513600000
               584,  0.967217900000,  0.827581300000,  0.001458533000
               585,  0.978600000000,  0.816300000000,  0.001400000000
               586,  0.989385600000,  0.804794700000,  0.001336667000
               587,  0.999548800000,  0.793082000000,  0.001270000000
               588,  1.009089200000,  0.781192000000,  0.001205000000
               589,  1.018006400000,  0.769154700000,  0.001146667000
               590,  1.026300000000,  0.757000000000,  0.001100000000
               591,  1.033982700000,  0.744754100000,  0.001068800000
               592,  1.040986000000,  0.732422400000,  0.001049400000
               593,  1.047188000000,  0.720003600000,  0.001035600000
               594,  1.052466700000,  0.707496500000,  0.001021200000
               595,  1.056700000000,  0.694900000000,  0.001000000000
               596,  1.059794400000,  0.682219200000,  0.000968640000
               597,  1.061799200000,  0.669471600000,  0.000929920000
               598,  1.062806800000,  0.656674400000,  0.000886880000
               599,  1.062909600000,  0.643844800000,  0.000842560000
               600,  1.062200000000,  0.631000000000,  0.000800000000
               601,  1.060735200000,  0.618155500000,  0.000760960000
               602,  1.058443600000,  0.605314400000,  0.000723680000
               603,  1.055224400000,  0.592475600000,  0.000685920000
               604,  1.050976800000,  0.579637900000,  0.000645440000
               605,  1.045600000000,  0.566800000000,  0.000600000000
               606,  1.039036900000,  0.553961100000,  0.000547866700
               607,  1.031360800000,  0.541137200000,  0.000491600000
               608,  1.022666200000,  0.528352800000,  0.000435400000
               609,  1.013047700000,  0.515632300000,  0.000383466700
               610,  1.002600000000,  0.503000000000,  0.000340000000
               611,  0.991367500000,  0.490468800000,  0.000307253300
               612,  0.979331400000,  0.478030400000,  0.000283160000
               613,  0.966491600000,  0.465677600000,  0.000265440000
               614,  0.952847900000,  0.453403200000,  0.000251813300
               615,  0.938400000000,  0.441200000000,  0.000240000000
               616,  0.923194000000,  0.429080000000,  0.000229546700
               617,  0.907244000000,  0.417036000000,  0.000220640000
               618,  0.890502000000,  0.405032000000,  0.000211960000
               619,  0.872920000000,  0.393032000000,  0.000202186700
               620,  0.854449900000,  0.381000000000,  0.000190000000
               621,  0.835084000000,  0.368918400000,  0.000174213300
               622,  0.814946000000,  0.356827200000,  0.000155640000
               623,  0.794186000000,  0.344776800000,  0.000135960000
               624,  0.772954000000,  0.332817600000,  0.000116853300
               625,  0.751400000000,  0.321000000000,  0.000100000000
               626,  0.729583600000,  0.309338100000,  0.000086133330
               627,  0.707588800000,  0.297850400000,  0.000074600000
               628,  0.685602200000,  0.286593600000,  0.000065000000
               629,  0.663810400000,  0.275624500000,  0.000056933330
               630,  0.642400000000,  0.265000000000,  0.000049999990
               631,  0.621514900000,  0.254763200000,  0.000044160000
               632,  0.601113800000,  0.244889600000,  0.000039480000
               633,  0.581105200000,  0.235334400000,  0.000035720000
               634,  0.561397700000,  0.226052800000,  0.000032640000
               635,  0.541900000000,  0.217000000000,  0.000030000000
               636,  0.522599500000,  0.208161600000,  0.000027653330
               637,  0.503546400000,  0.199548800000,  0.000025560000
               638,  0.484743600000,  0.191155200000,  0.000023640000
               639,  0.466193900000,  0.182974400000,  0.000021813330
               640,  0.447900000000,  0.175000000000,  0.000020000000
               641,  0.429861300000,  0.167223500000,  0.000018133330
               642,  0.412098000000,  0.159646400000,  0.000016200000
               643,  0.394644000000,  0.152277600000,  0.000014200000
               644,  0.377533300000,  0.145125900000,  0.000012133330
               645,  0.360800000000,  0.138200000000,  0.000010000000
               646,  0.344456300000,  0.131500300000,  0.000007733333
               647,  0.328516800000,  0.125024800000,  0.000005400000
               648,  0.313019200000,  0.118779200000,  0.000003200000
               649,  0.298001100000,  0.112769100000,  0.000001333333
               650,  0.283500000000,  0.107000000000,  0.000000000000
               651,  0.269544800000,  0.101476200000,  0.000000000000
               652,  0.256118400000,  0.096188640000,  0.000000000000
               653,  0.243189600000,  0.091122960000,  0.000000000000
               654,  0.230727200000,  0.086264850000,  0.000000000000
               655,  0.218700000000,  0.081600000000,  0.000000000000
               656,  0.207097100000,  0.077120640000,  0.000000000000
               657,  0.195923200000,  0.072825520000,  0.000000000000
               658,  0.185170800000,  0.068710080000,  0.000000000000
               659,  0.174832300000,  0.064769760000,  0.000000000000
               660,  0.164900000000,  0.061000000000,  0.000000000000
               661,  0.155366700000,  0.057396210000,  0.000000000000
               662,  0.146230000000,  0.053955040000,  0.000000000000
               663,  0.137490000000,  0.050673760000,  0.000000000000
               664,  0.129146700000,  0.047549650000,  0.000000000000
               665,  0.121200000000,  0.044580000000,  0.000000000000
               666,  0.113639700000,  0.041758720000,  0.000000000000
               667,  0.106465000000,  0.039084960000,  0.000000000000
               668,  0.099690440000,  0.036563840000,  0.000000000000
               669,  0.093330610000,  0.034200480000,  0.000000000000
               670,  0.087400000000,  0.032000000000,  0.000000000000
               671,  0.081900960000,  0.029962610000,  0.000000000000
               672,  0.076804280000,  0.028076640000,  0.000000000000
               673,  0.072077120000,  0.026329360000,  0.000000000000
               674,  0.067686640000,  0.024708050000,  0.000000000000
               675,  0.063600000000,  0.023200000000,  0.000000000000
               676,  0.059806850000,  0.021800770000,  0.000000000000
               677,  0.056282160000,  0.020501120000,  0.000000000000
               678,  0.052971040000,  0.019281080000,  0.000000000000
               679,  0.049818610000,  0.018120690000,  0.000000000000
               680,  0.046770000000,  0.017000000000,  0.000000000000
               681,  0.043784050000,  0.015903790000,  0.000000000000
               682,  0.040875360000,  0.014837180000,  0.000000000000
               683,  0.038072640000,  0.013810680000,  0.000000000000
               684,  0.035404610000,  0.012834780000,  0.000000000000
               685,  0.032900000000,  0.011920000000,  0.000000000000
               686,  0.030564190000,  0.011068310000,  0.000000000000
               687,  0.028380560000,  0.010273390000,  0.000000000000
               688,  0.026344840000,  0.009533311000,  0.000000000000
               689,  0.024452750000,  0.008846157000,  0.000000000000
               690,  0.022700000000,  0.008210000000,  0.000000000000
               691,  0.021084290000,  0.007623781000,  0.000000000000
               692,  0.019599880000,  0.007085424000,  0.000000000000
               693,  0.018237320000,  0.006591476000,  0.000000000000
               694,  0.016987170000,  0.006138485000,  0.000000000000
               695,  0.015840000000,  0.005723000000,  0.000000000000
               696,  0.014790640000,  0.005343059000,  0.000000000000
               697,  0.013831320000,  0.004995796000,  0.000000000000
               698,  0.012948680000,  0.004676404000,  0.000000000000
               699,  0.012129200000,  0.004380075000,  0.000000000000
               700,  0.011359160000,  0.004102000000,  0.000000000000
               701,  0.010629350000,  0.003838453000,  0.000000000000
               702,  0.009938846000,  0.003589099000,  0.000000000000
               703,  0.009288422000,  0.003354219000,  0.000000000000
               704,  0.008678854000,  0.003134093000,  0.000000000000
               705,  0.008110916000,  0.002929000000,  0.000000000000
               706,  0.007582388000,  0.002738139000,  0.000000000000
               707,  0.007088746000,  0.002559876000,  0.000000000000
               708,  0.006627313000,  0.002393244000,  0.000000000000
               709,  0.006195408000,  0.002237275000,  0.000000000000
               710,  0.005790346000,  0.002091000000,  0.000000000000
               711,  0.005409826000,  0.001953587000,  0.000000000000
               712,  0.005052583000,  0.001824580000,  0.000000000000
               713,  0.004717512000,  0.001703580000,  0.000000000000
               714,  0.004403507000,  0.001590187000,  0.000000000000
               715,  0.004109457000,  0.001484000000,  0.000000000000
               716,  0.003833913000,  0.001384496000,  0.000000000000
               717,  0.003575748000,  0.001291268000,  0.000000000000
               718,  0.003334342000,  0.001204092000,  0.000000000000
               719,  0.003109075000,  0.001122744000,  0.000000000000
               720,  0.002899327000,  0.001047000000,  0.000000000000
               721,  0.002704348000,  0.000976589600,  0.000000000000
               722,  0.002523020000,  0.000911108800,  0.000000000000
               723,  0.002354168000,  0.000850133200,  0.000000000000
               724,  0.002196616000,  0.000793238400,  0.000000000000
               725,  0.002049190000,  0.000740000000,  0.000000000000
               726,  0.001910960000,  0.000690082700,  0.000000000000
               727,  0.001781438000,  0.000643310000,  0.000000000000
               728,  0.001660110000,  0.000599496000,  0.000000000000
               729,  0.001546459000,  0.000558454700,  0.000000000000
               730,  0.001439971000,  0.000520000000,  0.000000000000
               731,  0.001340042000,  0.000483913600,  0.000000000000
               732,  0.001246275000,  0.000450052800,  0.000000000000
               733,  0.001158471000,  0.000418345200,  0.000000000000
               734,  0.001076430000,  0.000388718400,  0.000000000000
               735,  0.000999949300,  0.000361100000,  0.000000000000
               736,  0.000928735800,  0.000335383500,  0.000000000000
               737,  0.000862433200,  0.000311440400,  0.000000000000
               738,  0.000800750300,  0.000289165600,  0.000000000000
               739,  0.000743396000,  0.000268453900,  0.000000000000
               740,  0.000690078600,  0.000249200000,  0.000000000000
               741,  0.000640515600,  0.000231301900,  0.000000000000
               742,  0.000594502100,  0.000214685600,  0.000000000000
               743,  0.000551864600,  0.000199288400,  0.000000000000
               744,  0.000512429000,  0.000185047500,  0.000000000000
               745,  0.000476021300,  0.000171900000,  0.000000000000
               746,  0.000442453600,  0.000159778100,  0.000000000000
               747,  0.000411511700,  0.000148604400,  0.000000000000
               748,  0.000382981400,  0.000138301600,  0.000000000000
               749,  0.000356649100,  0.000128792500,  0.000000000000
               750,  0.000332301100,  0.000120000000,  0.000000000000
               751,  0.000309758600,  0.000111859500,  0.000000000000
               752,  0.000288887100,  0.000104322400,  0.000000000000
               753,  0.000269539400,  0.000097335600,  0.000000000000
               754,  0.000251568200,  0.000090845870,  0.000000000000
               755,  0.000234826100,  0.000084800000,  0.000000000000
               756,  0.000219171000,  0.000079146670,  0.000000000000
               757,  0.000204525800,  0.000073858000,  0.000000000000
               758,  0.000190840500,  0.000068916000,  0.000000000000
               759,  0.000178065400,  0.000064302670,  0.000000000000
               760,  0.000166150500,  0.000060000000,  0.000000000000
               761,  0.000155023600,  0.000055981870,  0.000000000000
               762,  0.000144621900,  0.000052225600,  0.000000000000
               763,  0.000134909800,  0.000048718400,  0.000000000000
               764,  0.000125852000,  0.000045447470,  0.000000000000
               765,  0.000117413000,  0.000042400000,  0.000000000000
               766,  0.000109551500,  0.000039561040,  0.000000000000
               767,  0.000102224500,  0.000036915120,  0.000000000000
               768,  0.000095394450,  0.000034448680,  0.000000000000
               769,  0.000089023900,  0.000032148160,  0.000000000000
               770,  0.000083075270,  0.000030000000,  0.000000000000
               771,  0.000077512690,  0.000027991250,  0.000000000000
               772,  0.000072313040,  0.000026113560,  0.000000000000
               773,  0.000067457780,  0.000024360240,  0.000000000000
               774,  0.000062928440,  0.000022724610,  0.000000000000
               775,  0.000058706520,  0.000021200000,  0.000000000000
               776,  0.000054770280,  0.000019778550,  0.000000000000
               777,  0.000051099180,  0.000018452850,  0.000000000000
               778,  0.000047676540,  0.000017216870,  0.000000000000
               779,  0.000044485670,  0.000016064590,  0.000000000000
               780,  0.000041509940,  0.000014990000,  0.000000000000
               781,  0.000038733240,  0.000013987280,  0.000000000000
               782,  0.000036142030,  0.000013051550,  0.000000000000
               783,  0.000033723520,  0.000012178180,  0.000000000000
               784,  0.000031464870,  0.000011362540,  0.000000000000
               785,  0.000029353260,  0.000010600000,  0.000000000000
               786,  0.000027375730,  0.000009885877,  0.000000000000
               787,  0.000025524330,  0.000009217304,  0.000000000000
               788,  0.000023793760,  0.000008592362,  0.000000000000
               789,  0.000022178700,  0.000008009133,  0.000000000000
               790,  0.000020673830,  0.000007465700,  0.000000000000
               791,  0.000019272260,  0.000006959567,  0.000000000000
               792,  0.000017966400,  0.000006487995,  0.000000000000
               793,  0.000016749910,  0.000006048699,  0.000000000000
               794,  0.000015616480,  0.000005639396,  0.000000000000
               795,  0.000014559770,  0.000005257800,  0.000000000000
               796,  0.000013573870,  0.000004901771,  0.000000000000
               797,  0.000012654360,  0.000004569720,  0.000000000000
               798,  0.000011797230,  0.000004260194,  0.000000000000
               799,  0.000010998440,  0.000003971739,  0.000000000000
               800,  0.000010253980,  0.000003702900,  0.000000000000
               801,  0.000009559646,  0.000003452163,  0.000000000000
               802,  0.000008912044,  0.000003218302,  0.000000000000
               803,  0.000008308358,  0.000003000300,  0.000000000000
               804,  0.000007745769,  0.000002797139,  0.000000000000
               805,  0.000007221456,  0.000002607800,  0.000000000000
               806,  0.000006732475,  0.000002431220,  0.000000000000
               807,  0.000006276423,  0.000002266531,  0.000000000000
               808,  0.000005851304,  0.000002113013,  0.000000000000
               809,  0.000005455118,  0.000001969943,  0.000000000000
               810,  0.000005085868,  0.000001836600,  0.000000000000
               811,  0.000004741466,  0.000001712230,  0.000000000000
               812,  0.000004420236,  0.000001596228,  0.000000000000
               813,  0.000004120783,  0.000001488090,  0.000000000000
               814,  0.000003841716,  0.000001387314,  0.000000000000
               815,  0.000003581652,  0.000001293400,  0.000000000000
               816,  0.000003339127,  0.000001205820,  0.000000000000
               817,  0.000003112949,  0.000001124143,  0.000000000000
               818,  0.000002902121,  0.000001048009,  0.000000000000
               819,  0.000002705645,  0.000000977058,  0.000000000000
               820,  0.000002522525,  0.000000910930,  0.000000000000
               821,  0.000002351726,  0.000000849251,  0.000000000000
               822,  0.000002192415,  0.000000791721,  0.000000000000
               823,  0.000002043902,  0.000000738090,  0.000000000000
               824,  0.000001905497,  0.000000688110,  0.000000000000
               825,  0.000001776509,  0.000000641530,  0.000000000000
               826,  0.000001656215,  0.000000598090,  0.000000000000
               827,  0.000001544022,  0.000000557575,  0.000000000000
               828,  0.000001439440,  0.000000519808,  0.000000000000
               829,  0.000001341977,  0.000000484612,  0.000000000000
               830,  0.000001251141,  0.000000451810,  0.000000000000];

    case '1964_full'
        
        cmf = [360,  0.000000122200,  0.000000013398,  0.000000535027
               361,  0.000000185138,  0.000000020294,  0.000000810720
               362,  0.000000278830,  0.000000030560,  0.000001221200
               363,  0.000000417470,  0.000000045740,  0.000001828700
               364,  0.000000621330,  0.000000068050,  0.000002722200
               365,  0.000000919270,  0.000000100650,  0.000004028300
               366,  0.000001351980,  0.000000147980,  0.000005925700
               367,  0.000001976540,  0.000000216270,  0.000008665100
               368,  0.000002872500,  0.000000314200,  0.000012596000
               369,  0.000004149500,  0.000000453700,  0.000018201000
               370,  0.000005958600,  0.000000651100,  0.000026143700
               371,  0.000008505600,  0.000000928800,  0.000037330000
               372,  0.000012068600,  0.000001317500,  0.000052987000
               373,  0.000017022600,  0.000001857200,  0.000074764000
               374,  0.000023868000,  0.000002602000,  0.000104870000
               375,  0.000033266000,  0.000003625000,  0.000146220000
               376,  0.000046087000,  0.000005019000,  0.000202660000
               377,  0.000063472000,  0.000006907000,  0.000279230000
               378,  0.000086892000,  0.000009449000,  0.000382450000
               379,  0.000118246000,  0.000012848000,  0.000520720000
               380,  0.000159952000,  0.000017364000,  0.000704776000
               381,  0.000215080000,  0.000023327000,  0.000948230000
               382,  0.000287490000,  0.000031150000,  0.001268200000
               383,  0.000381990000,  0.000041350000,  0.001686100000
               384,  0.000504550000,  0.000054560000,  0.002228500000
               385,  0.000662440000,  0.000071560000,  0.002927800000
               386,  0.000864500000,  0.000093300000,  0.003823700000
               387,  0.001121500000,  0.000120870000,  0.004964200000
               388,  0.001446160000,  0.000155640000,  0.006406700000
               389,  0.001853590000,  0.000199200000,  0.008219300000
               390,  0.002361600000,  0.000253400000,  0.010482200000
               391,  0.002990600000,  0.000320200000,  0.013289000000
               392,  0.003764500000,  0.000402400000,  0.016747000000
               393,  0.004710200000,  0.000502300000,  0.020980000000
               394,  0.005858100000,  0.000623200000,  0.026127000000
               395,  0.007242300000,  0.000768500000,  0.032344000000
               396,  0.008899600000,  0.000941700000,  0.039802000000
               397,  0.010870900000,  0.001147800000,  0.048691000000
               398,  0.013198900000,  0.001390300000,  0.059210000000
               399,  0.015929200000,  0.001674000000,  0.071576000000
               400,  0.019109700000,  0.002004400000,  0.086010900000
               401,  0.022788000000,  0.002386000000,  0.102740000000
               402,  0.027011000000,  0.002822000000,  0.122000000000
               403,  0.031829000000,  0.003319000000,  0.144020000000
               404,  0.037278000000,  0.003880000000,  0.168990000000
               405,  0.043400000000,  0.004509000000,  0.197120000000
               406,  0.050223000000,  0.005209000000,  0.228570000000
               407,  0.057764000000,  0.005985000000,  0.263470000000
               408,  0.066038000000,  0.006833000000,  0.301900000000
               409,  0.075033000000,  0.007757000000,  0.343870000000
               410,  0.084736000000,  0.008756000000,  0.389366000000
               411,  0.095041000000,  0.009816000000,  0.437970000000
               412,  0.105836000000,  0.010918000000,  0.489220000000
               413,  0.117066000000,  0.012058000000,  0.542900000000
               414,  0.128682000000,  0.013237000000,  0.598810000000
               415,  0.140638000000,  0.014456000000,  0.656760000000
               416,  0.152893000000,  0.015717000000,  0.716580000000
               417,  0.165416000000,  0.017025000000,  0.778120000000
               418,  0.178191000000,  0.018399000000,  0.841310000000
               419,  0.191214000000,  0.019848000000,  0.906110000000
               420,  0.204492000000,  0.021391000000,  0.972542000000
               421,  0.217650000000,  0.022992000000,  1.038900000000
               422,  0.230267000000,  0.024598000000,  1.103100000000
               423,  0.242311000000,  0.026213000000,  1.165100000000
               424,  0.253793000000,  0.027841000000,  1.224900000000
               425,  0.264737000000,  0.029497000000,  1.282500000000
               426,  0.275195000000,  0.031195000000,  1.338200000000
               427,  0.285301000000,  0.032927000000,  1.392600000000
               428,  0.295143000000,  0.034738000000,  1.446100000000
               429,  0.304869000000,  0.036654000000,  1.499400000000
               430,  0.314679000000,  0.038676000000,  1.553480000000
               431,  0.324355000000,  0.040792000000,  1.607200000000
               432,  0.333570000000,  0.042946000000,  1.658900000000
               433,  0.342243000000,  0.045114000000,  1.708200000000
               434,  0.350312000000,  0.047333000000,  1.754800000000
               435,  0.357719000000,  0.049602000000,  1.798500000000
               436,  0.364482000000,  0.051934000000,  1.839200000000
               437,  0.370493000000,  0.054337000000,  1.876600000000
               438,  0.375727000000,  0.056822000000,  1.910500000000
               439,  0.380158000000,  0.059399000000,  1.940800000000
               440,  0.383734000000,  0.062077000000,  1.967280000000
               441,  0.386327000000,  0.064737000000,  1.989100000000
               442,  0.387858000000,  0.067285000000,  2.005700000000
               443,  0.388396000000,  0.069764000000,  2.017400000000
               444,  0.387978000000,  0.072218000000,  2.024400000000
               445,  0.386726000000,  0.074704000000,  2.027300000000
               446,  0.384696000000,  0.077272000000,  2.026400000000
               447,  0.382006000000,  0.079979000000,  2.022300000000
               448,  0.378709000000,  0.082874000000,  2.015300000000
               449,  0.374915000000,  0.086000000000,  2.006000000000
               450,  0.370702000000,  0.089456000000,  1.994800000000
               451,  0.366089000000,  0.092947000000,  1.981400000000
               452,  0.361045000000,  0.096275000000,  1.965300000000
               453,  0.355518000000,  0.099535000000,  1.946400000000
               454,  0.349486000000,  0.102829000000,  1.924800000000
               455,  0.342957000000,  0.106256000000,  1.900700000000
               456,  0.335893000000,  0.109901000000,  1.874100000000
               457,  0.328284000000,  0.113835000000,  1.845100000000
               458,  0.320150000000,  0.118167000000,  1.813900000000
               459,  0.311475000000,  0.122932000000,  1.780600000000
               460,  0.302273000000,  0.128201000000,  1.745370000000
               461,  0.292858000000,  0.133457000000,  1.709100000000
               462,  0.283502000000,  0.138323000000,  1.672300000000
               463,  0.274044000000,  0.143042000000,  1.634700000000
               464,  0.264263000000,  0.147787000000,  1.595600000000
               465,  0.254085000000,  0.152761000000,  1.554900000000
               466,  0.243392000000,  0.158102000000,  1.512200000000
               467,  0.232187000000,  0.163941000000,  1.467300000000
               468,  0.220488000000,  0.170362000000,  1.419900000000
               469,  0.208198000000,  0.177425000000,  1.370000000000
               470,  0.195618000000,  0.185190000000,  1.317560000000
               471,  0.183034000000,  0.193025000000,  1.262400000000
               472,  0.170222000000,  0.200313000000,  1.205000000000
               473,  0.157348000000,  0.207156000000,  1.146600000000
               474,  0.144650000000,  0.213644000000,  1.088000000000
               475,  0.132349000000,  0.219940000000,  1.030200000000
               476,  0.120584000000,  0.226170000000,  0.973830000000
               477,  0.109456000000,  0.232467000000,  0.919430000000
               478,  0.099042000000,  0.239025000000,  0.867460000000
               479,  0.089388000000,  0.245997000000,  0.818280000000
               480,  0.080507000000,  0.253589000000,  0.772125000000
               481,  0.072034000000,  0.261876000000,  0.728290000000
               482,  0.063710000000,  0.270643000000,  0.686040000000
               483,  0.055694000000,  0.279645000000,  0.645530000000
               484,  0.048117000000,  0.288694000000,  0.606850000000
               485,  0.041072000000,  0.297665000000,  0.570060000000
               486,  0.034642000000,  0.306469000000,  0.535220000000
               487,  0.028896000000,  0.315035000000,  0.502340000000
               488,  0.023876000000,  0.323335000000,  0.471400000000
               489,  0.019628000000,  0.331366000000,  0.442390000000
               490,  0.016172000000,  0.339133000000,  0.415254000000
               491,  0.013300000000,  0.347860000000,  0.390024000000
               492,  0.010759000000,  0.358326000000,  0.366399000000
               493,  0.008542000000,  0.370001000000,  0.344015000000
               494,  0.006661000000,  0.382464000000,  0.322689000000
               495,  0.005132000000,  0.395379000000,  0.302356000000
               496,  0.003982000000,  0.408482000000,  0.283036000000
               497,  0.003239000000,  0.421588000000,  0.264816000000
               498,  0.002934000000,  0.434619000000,  0.247848000000
               499,  0.003114000000,  0.447601000000,  0.232318000000
               500,  0.003816000000,  0.460777000000,  0.218502000000
               501,  0.005095000000,  0.474340000000,  0.205851000000
               502,  0.006936000000,  0.488200000000,  0.193596000000
               503,  0.009299000000,  0.502340000000,  0.181736000000
               504,  0.012147000000,  0.516740000000,  0.170281000000
               505,  0.015444000000,  0.531360000000,  0.159249000000
               506,  0.019156000000,  0.546190000000,  0.148673000000
               507,  0.023250000000,  0.561180000000,  0.138609000000
               508,  0.027690000000,  0.576290000000,  0.129096000000
               509,  0.032444000000,  0.591500000000,  0.120215000000
               510,  0.037465000000,  0.606741000000,  0.112044000000
               511,  0.042956000000,  0.622150000000,  0.104710000000
               512,  0.049114000000,  0.637830000000,  0.098196000000
               513,  0.055920000000,  0.653710000000,  0.092361000000
               514,  0.063349000000,  0.669680000000,  0.087088000000
               515,  0.071358000000,  0.685660000000,  0.082248000000
               516,  0.079901000000,  0.701550000000,  0.077744000000
               517,  0.088909000000,  0.717230000000,  0.073456000000
               518,  0.098293000000,  0.732570000000,  0.069268000000
               519,  0.107949000000,  0.747460000000,  0.065060000000
               520,  0.117749000000,  0.761757000000,  0.060709000000
               521,  0.127839000000,  0.775340000000,  0.056457000000
               522,  0.138450000000,  0.788220000000,  0.052609000000
               523,  0.149516000000,  0.800460000000,  0.049122000000
               524,  0.161041000000,  0.812140000000,  0.045954000000
               525,  0.172953000000,  0.823330000000,  0.043050000000
               526,  0.185209000000,  0.834120000000,  0.040368000000
               527,  0.197755000000,  0.844600000000,  0.037839000000
               528,  0.210538000000,  0.854870000000,  0.035384000000
               529,  0.223460000000,  0.865040000000,  0.032949000000
               530,  0.236491000000,  0.875211000000,  0.030451000000
               531,  0.249633000000,  0.885370000000,  0.028029000000
               532,  0.262972000000,  0.895370000000,  0.025862000000
               533,  0.276515000000,  0.905150000000,  0.023920000000
               534,  0.290269000000,  0.914650000000,  0.022174000000
               535,  0.304213000000,  0.923810000000,  0.020584000000
               536,  0.318361000000,  0.932550000000,  0.019127000000
               537,  0.332705000000,  0.940810000000,  0.017740000000
               538,  0.347232000000,  0.948520000000,  0.016403000000
               539,  0.361926000000,  0.955600000000,  0.015064000000
               540,  0.376772000000,  0.961988000000,  0.013676000000
               541,  0.391683000000,  0.967540000000,  0.012308000000
               542,  0.406594000000,  0.972230000000,  0.011056000000
               543,  0.421539000000,  0.976170000000,  0.009915000000
               544,  0.436517000000,  0.979460000000,  0.008872000000
               545,  0.451584000000,  0.982200000000,  0.007918000000
               546,  0.466782000000,  0.984520000000,  0.007030000000
               547,  0.482147000000,  0.986520000000,  0.006223000000
               548,  0.497738000000,  0.988320000000,  0.005453000000
               549,  0.513606000000,  0.990020000000,  0.004714000000
               550,  0.529826000000,  0.991761000000,  0.003988000000
               551,  0.546440000000,  0.993530000000,  0.003289000000
               552,  0.563426000000,  0.995230000000,  0.002646000000
               553,  0.580726000000,  0.996770000000,  0.002063000000
               554,  0.598290000000,  0.998090000000,  0.001533000000
               555,  0.616053000000,  0.999110000000,  0.001091000000
               556,  0.633948000000,  0.999770000000,  0.000711000000
               557,  0.651901000000,  1.000000000000,  0.000407000000
               558,  0.669824000000,  0.999710000000,  0.000184000000
               559,  0.687632000000,  0.998850000000,  0.000047000000
               560,  0.705224000000,  0.997340000000,  0.000000000000
               561,  0.722773000000,  0.995260000000,  0.000000000000
               562,  0.740483000000,  0.992740000000,  0.000000000000
               563,  0.758273000000,  0.989750000000,  0.000000000000
               564,  0.776083000000,  0.986300000000,  0.000000000000
               565,  0.793832000000,  0.982380000000,  0.000000000000
               566,  0.811436000000,  0.977980000000,  0.000000000000
               567,  0.828822000000,  0.973110000000,  0.000000000000
               568,  0.845879000000,  0.967740000000,  0.000000000000
               569,  0.862525000000,  0.961890000000,  0.000000000000
               570,  0.878655000000,  0.955552000000,  0.000000000000
               571,  0.894208000000,  0.948601000000,  0.000000000000
               572,  0.909206000000,  0.940981000000,  0.000000000000
               573,  0.923672000000,  0.932798000000,  0.000000000000
               574,  0.937638000000,  0.924158000000,  0.000000000000
               575,  0.951162000000,  0.915175000000,  0.000000000000
               576,  0.964283000000,  0.905954000000,  0.000000000000
               577,  0.977068000000,  0.896608000000,  0.000000000000
               578,  0.989590000000,  0.887249000000,  0.000000000000
               579,  1.001910000000,  0.877986000000,  0.000000000000
               580,  1.014160000000,  0.868934000000,  0.000000000000
               581,  1.026500000000,  0.860164000000,  0.000000000000
               582,  1.038800000000,  0.851519000000,  0.000000000000
               583,  1.051000000000,  0.842963000000,  0.000000000000
               584,  1.062900000000,  0.834393000000,  0.000000000000
               585,  1.074300000000,  0.825623000000,  0.000000000000
               586,  1.085200000000,  0.816764000000,  0.000000000000
               587,  1.095200000000,  0.807544000000,  0.000000000000
               588,  1.104200000000,  0.797947000000,  0.000000000000
               589,  1.112000000000,  0.787893000000,  0.000000000000
               590,  1.118520000000,  0.777405000000,  0.000000000000
               591,  1.123800000000,  0.766490000000,  0.000000000000
               592,  1.128000000000,  0.755309000000,  0.000000000000
               593,  1.131100000000,  0.743845000000,  0.000000000000
               594,  1.133200000000,  0.732190000000,  0.000000000000
               595,  1.134300000000,  0.720353000000,  0.000000000000
               596,  1.134300000000,  0.708281000000,  0.000000000000
               597,  1.133300000000,  0.696055000000,  0.000000000000
               598,  1.131200000000,  0.683621000000,  0.000000000000
               599,  1.128100000000,  0.671048000000,  0.000000000000
               600,  1.123990000000,  0.658341000000,  0.000000000000
               601,  1.118900000000,  0.645545000000,  0.000000000000
               602,  1.112900000000,  0.632718000000,  0.000000000000
               603,  1.105900000000,  0.619815000000,  0.000000000000
               604,  1.098000000000,  0.606887000000,  0.000000000000
               605,  1.089100000000,  0.593878000000,  0.000000000000
               606,  1.079200000000,  0.580781000000,  0.000000000000
               607,  1.068400000000,  0.567653000000,  0.000000000000
               608,  1.056700000000,  0.554490000000,  0.000000000000
               609,  1.044000000000,  0.541228000000,  0.000000000000
               610,  1.030480000000,  0.527963000000,  0.000000000000
               611,  1.016000000000,  0.514634000000,  0.000000000000
               612,  1.000800000000,  0.501363000000,  0.000000000000
               613,  0.984790000000,  0.488124000000,  0.000000000000
               614,  0.968080000000,  0.474935000000,  0.000000000000
               615,  0.950740000000,  0.461834000000,  0.000000000000
               616,  0.932800000000,  0.448823000000,  0.000000000000
               617,  0.914340000000,  0.435917000000,  0.000000000000
               618,  0.895390000000,  0.423153000000,  0.000000000000
               619,  0.876030000000,  0.410526000000,  0.000000000000
               620,  0.856297000000,  0.398057000000,  0.000000000000
               621,  0.836350000000,  0.385835000000,  0.000000000000
               622,  0.816290000000,  0.373951000000,  0.000000000000
               623,  0.796050000000,  0.362311000000,  0.000000000000
               624,  0.775610000000,  0.350863000000,  0.000000000000
               625,  0.754930000000,  0.339554000000,  0.000000000000
               626,  0.733990000000,  0.328309000000,  0.000000000000
               627,  0.712780000000,  0.317118000000,  0.000000000000
               628,  0.691290000000,  0.305936000000,  0.000000000000
               629,  0.669520000000,  0.294737000000,  0.000000000000
               630,  0.647467000000,  0.283493000000,  0.000000000000
               631,  0.625110000000,  0.272222000000,  0.000000000000
               632,  0.602520000000,  0.260990000000,  0.000000000000
               633,  0.579890000000,  0.249877000000,  0.000000000000
               634,  0.557370000000,  0.238946000000,  0.000000000000
               635,  0.535110000000,  0.228254000000,  0.000000000000
               636,  0.513240000000,  0.217853000000,  0.000000000000
               637,  0.491860000000,  0.207780000000,  0.000000000000
               638,  0.471080000000,  0.198072000000,  0.000000000000
               639,  0.450960000000,  0.188748000000,  0.000000000000
               640,  0.431567000000,  0.179828000000,  0.000000000000
               641,  0.412870000000,  0.171285000000,  0.000000000000
               642,  0.394750000000,  0.163059000000,  0.000000000000
               643,  0.377210000000,  0.155151000000,  0.000000000000
               644,  0.360190000000,  0.147535000000,  0.000000000000
               645,  0.343690000000,  0.140211000000,  0.000000000000
               646,  0.327690000000,  0.133170000000,  0.000000000000
               647,  0.312170000000,  0.126400000000,  0.000000000000
               648,  0.297110000000,  0.119892000000,  0.000000000000
               649,  0.282500000000,  0.113640000000,  0.000000000000
               650,  0.268329000000,  0.107633000000,  0.000000000000
               651,  0.254590000000,  0.101870000000,  0.000000000000
               652,  0.241300000000,  0.096347000000,  0.000000000000
               653,  0.228480000000,  0.091063000000,  0.000000000000
               654,  0.216140000000,  0.086010000000,  0.000000000000
               655,  0.204300000000,  0.081187000000,  0.000000000000
               656,  0.192950000000,  0.076583000000,  0.000000000000
               657,  0.182110000000,  0.072198000000,  0.000000000000
               658,  0.171770000000,  0.068024000000,  0.000000000000
               659,  0.161920000000,  0.064052000000,  0.000000000000
               660,  0.152568000000,  0.060281000000,  0.000000000000
               661,  0.143670000000,  0.056697000000,  0.000000000000
               662,  0.135200000000,  0.053292000000,  0.000000000000
               663,  0.127130000000,  0.050059000000,  0.000000000000
               664,  0.119480000000,  0.046998000000,  0.000000000000
               665,  0.112210000000,  0.044096000000,  0.000000000000
               666,  0.105310000000,  0.041345000000,  0.000000000000
               667,  0.098786000000,  0.038750700000,  0.000000000000
               668,  0.092610000000,  0.036297800000,  0.000000000000
               669,  0.086773000000,  0.033983200000,  0.000000000000
               670,  0.081260600000,  0.031800400000,  0.000000000000
               671,  0.076048000000,  0.029739500000,  0.000000000000
               672,  0.071114000000,  0.027791800000,  0.000000000000
               673,  0.066454000000,  0.025955100000,  0.000000000000
               674,  0.062062000000,  0.024226300000,  0.000000000000
               675,  0.057930000000,  0.022601700000,  0.000000000000
               676,  0.054050000000,  0.021077900000,  0.000000000000
               677,  0.050412000000,  0.019650500000,  0.000000000000
               678,  0.047006000000,  0.018315300000,  0.000000000000
               679,  0.043823000000,  0.017068600000,  0.000000000000
               680,  0.040850800000,  0.015905100000,  0.000000000000
               681,  0.038072000000,  0.014818300000,  0.000000000000
               682,  0.035468000000,  0.013800800000,  0.000000000000
               683,  0.033031000000,  0.012849500000,  0.000000000000
               684,  0.030753000000,  0.011960700000,  0.000000000000
               685,  0.028623000000,  0.011130300000,  0.000000000000
               686,  0.026635000000,  0.010355500000,  0.000000000000
               687,  0.024781000000,  0.009633200000,  0.000000000000
               688,  0.023052000000,  0.008959900000,  0.000000000000
               689,  0.021441000000,  0.008332400000,  0.000000000000
               690,  0.019941300000,  0.007748800000,  0.000000000000
               691,  0.018544000000,  0.007204600000,  0.000000000000
               692,  0.017241000000,  0.006697500000,  0.000000000000
               693,  0.016027000000,  0.006225100000,  0.000000000000
               694,  0.014896000000,  0.005785000000,  0.000000000000
               695,  0.013842000000,  0.005375100000,  0.000000000000
               696,  0.012862000000,  0.004994100000,  0.000000000000
               697,  0.011949000000,  0.004639200000,  0.000000000000
               698,  0.011100000000,  0.004309300000,  0.000000000000
               699,  0.010311000000,  0.004002800000,  0.000000000000
               700,  0.009576880000,  0.003717740000,  0.000000000000
               701,  0.008894000000,  0.003452620000,  0.000000000000
               702,  0.008258100000,  0.003205830000,  0.000000000000
               703,  0.007666400000,  0.002976230000,  0.000000000000
               704,  0.007116300000,  0.002762810000,  0.000000000000
               705,  0.006605200000,  0.002564560000,  0.000000000000
               706,  0.006130600000,  0.002380480000,  0.000000000000
               707,  0.005690300000,  0.002209710000,  0.000000000000
               708,  0.005281900000,  0.002051320000,  0.000000000000
               709,  0.004903300000,  0.001904490000,  0.000000000000
               710,  0.004552630000,  0.001768470000,  0.000000000000
               711,  0.004227500000,  0.001642360000,  0.000000000000
               712,  0.003925800000,  0.001525350000,  0.000000000000
               713,  0.003645700000,  0.001416720000,  0.000000000000
               714,  0.003385900000,  0.001315950000,  0.000000000000
               715,  0.003144700000,  0.001222390000,  0.000000000000
               716,  0.002920800000,  0.001135550000,  0.000000000000
               717,  0.002713000000,  0.001054940000,  0.000000000000
               718,  0.002520200000,  0.000980140000,  0.000000000000
               719,  0.002341100000,  0.000910660000,  0.000000000000
               720,  0.002174960000,  0.000846190000,  0.000000000000
               721,  0.002020600000,  0.000786290000,  0.000000000000
               722,  0.001877300000,  0.000730680000,  0.000000000000
               723,  0.001744100000,  0.000678990000,  0.000000000000
               724,  0.001620500000,  0.000631010000,  0.000000000000
               725,  0.001505700000,  0.000586440000,  0.000000000000
               726,  0.001399200000,  0.000545110000,  0.000000000000
               727,  0.001300400000,  0.000506720000,  0.000000000000
               728,  0.001208700000,  0.000471110000,  0.000000000000
               729,  0.001123600000,  0.000438050000,  0.000000000000
               730,  0.001044760000,  0.000407410000,  0.000000000000
               731,  0.000971560000,  0.000378962000,  0.000000000000
               732,  0.000903600000,  0.000352543000,  0.000000000000
               733,  0.000840480000,  0.000328001000,  0.000000000000
               734,  0.000781870000,  0.000305208000,  0.000000000000
               735,  0.000727450000,  0.000284041000,  0.000000000000
               736,  0.000676900000,  0.000264375000,  0.000000000000
               737,  0.000629960000,  0.000246109000,  0.000000000000
               738,  0.000586370000,  0.000229143000,  0.000000000000
               739,  0.000545870000,  0.000213376000,  0.000000000000
               740,  0.000508258000,  0.000198730000,  0.000000000000
               741,  0.000473300000,  0.000185115000,  0.000000000000
               742,  0.000440800000,  0.000172454000,  0.000000000000
               743,  0.000410580000,  0.000160678000,  0.000000000000
               744,  0.000382490000,  0.000149730000,  0.000000000000
               745,  0.000356380000,  0.000139550000,  0.000000000000
               746,  0.000332110000,  0.000130086000,  0.000000000000
               747,  0.000309550000,  0.000121290000,  0.000000000000
               748,  0.000288580000,  0.000113106000,  0.000000000000
               749,  0.000269090000,  0.000105501000,  0.000000000000
               750,  0.000250969000,  0.000098428000,  0.000000000000
               751,  0.000234130000,  0.000091853000,  0.000000000000
               752,  0.000218470000,  0.000085738000,  0.000000000000
               753,  0.000203910000,  0.000080048000,  0.000000000000
               754,  0.000190350000,  0.000074751000,  0.000000000000
               755,  0.000177730000,  0.000069819000,  0.000000000000
               756,  0.000165970000,  0.000065222000,  0.000000000000
               757,  0.000155020000,  0.000060939000,  0.000000000000
               758,  0.000144800000,  0.000056942000,  0.000000000000
               759,  0.000135280000,  0.000053217000,  0.000000000000
               760,  0.000126390000,  0.000049737000,  0.000000000000
               761,  0.000118100000,  0.000046491000,  0.000000000000
               762,  0.000110370000,  0.000043464000,  0.000000000000
               763,  0.000103150000,  0.000040635000,  0.000000000000
               764,  0.000096427000,  0.000038000000,  0.000000000000
               765,  0.000090151000,  0.000035540500,  0.000000000000
               766,  0.000084294000,  0.000033244800,  0.000000000000
               767,  0.000078830000,  0.000031100600,  0.000000000000
               768,  0.000073729000,  0.000029099000,  0.000000000000
               769,  0.000068969000,  0.000027230700,  0.000000000000
               770,  0.000064525800,  0.000025486000,  0.000000000000
               771,  0.000060376000,  0.000023856100,  0.000000000000
               772,  0.000056500000,  0.000022333200,  0.000000000000
               773,  0.000052880000,  0.000020910400,  0.000000000000
               774,  0.000049498000,  0.000019580800,  0.000000000000
               775,  0.000046339000,  0.000018338400,  0.000000000000
               776,  0.000043389000,  0.000017177700,  0.000000000000
               777,  0.000040634000,  0.000016093400,  0.000000000000
               778,  0.000038060000,  0.000015080000,  0.000000000000
               779,  0.000035657000,  0.000014133600,  0.000000000000
               780,  0.000033411700,  0.000013249000,  0.000000000000
               781,  0.000031315000,  0.000012422600,  0.000000000000
               782,  0.000029355000,  0.000011649900,  0.000000000000
               783,  0.000027524000,  0.000010927700,  0.000000000000
               784,  0.000025811000,  0.000010251900,  0.000000000000
               785,  0.000024209000,  0.000009619600,  0.000000000000
               786,  0.000022711000,  0.000009028100,  0.000000000000
               787,  0.000021308000,  0.000008474000,  0.000000000000
               788,  0.000019994000,  0.000007954800,  0.000000000000
               789,  0.000018764000,  0.000007468600,  0.000000000000
               790,  0.000017611500,  0.000007012800,  0.000000000000
               791,  0.000016532000,  0.000006585800,  0.000000000000
               792,  0.000015521000,  0.000006185700,  0.000000000000
               793,  0.000014574000,  0.000005810700,  0.000000000000
               794,  0.000013686000,  0.000005459000,  0.000000000000
               795,  0.000012855000,  0.000005129800,  0.000000000000
               796,  0.000012075000,  0.000004820600,  0.000000000000
               797,  0.000011345000,  0.000004531200,  0.000000000000
               798,  0.000010659000,  0.000004259100,  0.000000000000
               799,  0.000010017000,  0.000004004200,  0.000000000000
               800,  0.000009413630,  0.000003764730,  0.000000000000
               801,  0.000008847900,  0.000003539950,  0.000000000000
               802,  0.000008317100,  0.000003329140,  0.000000000000
               803,  0.000007819000,  0.000003131150,  0.000000000000
               804,  0.000007351600,  0.000002945290,  0.000000000000
               805,  0.000006913000,  0.000002770810,  0.000000000000
               806,  0.000006501500,  0.000002607050,  0.000000000000
               807,  0.000006115300,  0.000002453290,  0.000000000000
               808,  0.000005752900,  0.000002308940,  0.000000000000
               809,  0.000005412700,  0.000002173380,  0.000000000000
               810,  0.000005093470,  0.000002046130,  0.000000000000
               811,  0.000004793800,  0.000001926620,  0.000000000000
               812,  0.000004512500,  0.000001814400,  0.000000000000
               813,  0.000004248300,  0.000001708950,  0.000000000000
               814,  0.000004000200,  0.000001609880,  0.000000000000
               815,  0.000003767100,  0.000001516770,  0.000000000000
               816,  0.000003548000,  0.000001429210,  0.000000000000
               817,  0.000003342100,  0.000001346860,  0.000000000000
               818,  0.000003148500,  0.000001269450,  0.000000000000
               819,  0.000002966500,  0.000001196620,  0.000000000000
               820,  0.000002795310,  0.000001128090,  0.000000000000
               821,  0.000002634500,  0.000001063680,  0.000000000000
               822,  0.000002483400,  0.000001003130,  0.000000000000
               823,  0.000002341400,  0.000000946220,  0.000000000000
               824,  0.000002207800,  0.000000892630,  0.000000000000
               825,  0.000002082000,  0.000000842160,  0.000000000000
               826,  0.000001963600,  0.000000794640,  0.000000000000
               827,  0.000001851900,  0.000000749780,  0.000000000000
               828,  0.000001746500,  0.000000707440,  0.000000000000
               829,  0.000001647100,  0.000000667480,  0.000000000000
               830,  0.000001553140,  0.000000629700,  0.000000000000];

    otherwise
        
        error('colorMatchFcn:unrecognizedMatchFcn', ...
              'Unrecognized color match function.')
          
end

lambda = cmf(:, 1)';
xFcn = cmf(:, 2)';
yFcn = cmf(:, 3)';
zFcn = cmf(:, 4)';
