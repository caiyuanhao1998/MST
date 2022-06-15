function createfigure(X1, YMatrix1, Corr)
%CREATEFIGURE(X1, YMatrix1)
%  X1:  x 数据的向量
%  YMATRIX1:  y 数据的矩阵

%  由 MATLAB 于 19-Feb-2022 11:12:35 自动生成

% 创建 figure
figure1 = figure('PaperOrientation','landscape',...
    'PaperSize',[29.69999902 20.99999864]);

% 创建 axes
axes1 = axes('Parent',figure1,'Position',[0.13 0.11 0.385625 0.815]);
hold(axes1,'on');

% 使用 plot 的矩阵输入创建多行
plot1 = plot(X1,YMatrix1,'MarkerSize',16,'Marker','.','LineWidth',2.5,...
    'Parent',axes1);
set(plot1(1),'DisplayName',' Ground Truth','Color',[1 0 0]);
set(plot1(2),'DisplayName',' DeSCI, corr: '+string(roundn(Corr(1),-04)),...
    'Color',[1 0 1]);
set(plot1(3),'DisplayName',' GAP-TV, corr: '+string(roundn(Corr(2),-4)),...
    'Color',[0 0 1]);
set(plot1(4),'DisplayName',' HSSP, corr: '+string(roundn(Corr(3),-4)),'Color',[0 0 0]);
set(plot1(5),'DisplayName',' \lambda-Net, corr: '+string(roundn(Corr(4),-4)),...
    'Color',[0 1 1]);
set(plot1(6),'DisplayName',' TSA-Net, corr: '+string(roundn(Corr(5),-4)),'Color',[0.5 0.16 0.16]);
set(plot1(7),'DisplayName',' TwIST, corr: '+string(roundn(Corr(6),-4)),'Color',[0.01 0.66 0.62]);
set(plot1(8),'DisplayName',' GAP-Net, corr: '+string(roundn(Corr(7),-4)),'Color',[0.66 0.01 0.62]);
set(plot1(9),'DisplayName',' DGSMP, corr: '+string(roundn(Corr(8),-4)),'Color',[1 1 0]);
set(plot1(10),'DisplayName',' MST-L, corr: '+string(roundn(Corr(9),-4)),'Color',[0 0.5 0.5]);



% 取消以下行的注释以保留坐标区的 Y 范围
ylim(axes1,[0 1]);
box(axes1,'on');
hold(axes1,'off');
% 设置其余坐标区属性
set(axes1,'FontName','Arial','FontSize',22,'LineWidth',3.5);

% 创建 ylabel
ylabel('Density','FontSize',28,'FontName','Arial');

% 创建 xlabel
xlabel('Wavelength (nm)','FontSize',28,'FontName','Arial');
% 创建 legend
legend1 = legend(axes1,'show');
set(legend1,...
    'Position',[0.320670220276361 0.124725505873052 0.187369795342287 0.36915888702758],...
    'FontSize',22,...
    'EdgeColor',[1 1 1]);

