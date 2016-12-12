A = [1 2 3 4 5 6 7 8 9 10 11 12 13];
B = [1 2 3 4 5 6 7 8 9 10 11 12 13];

filename = 'Prediction1.csv';
%image = im2double(imread(filename));

M = csvread(filename);

[m,n] = size(M)

D = zeros(13,13);
for i=1:length(A)
    for j=1:length(B)
        D(j,i) = M(i,j);
    end
end


%mat = rand(5);           %# A 5-by-5 matrix of random values from 0 to 1
imagesc(D);            %# Create a colored plot of the matrix values
colormap(flipud(gray));  %# Change the colormap to gray (so higher values are
                         %#   black and lower values are white)

textStrings = num2str(D(:),'%0.2f');  %# Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding
[x,y] = meshgrid(1:13);   %# Create x and y coordinates for the strings
hStrings = text(x(:),y(:),textStrings(:),...      %# Plot the strings
                'HorizontalAlignment','center');
midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
textColors = repmat(D(:) > midValue,1,3);  %# Choose white or black for the
                                             %#   text color of the strings so
                                             %#   they can be easily seen over
                                             %#   the background color
set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors

set(gca,'XTick',1:13,...                         %# Change the axes tick marks
        'XTickLabel',{'1','2','3','4','5', '6','7','8','9','10','11','12','13'},...  %#   and tick labels
        'YTick',1:13,...
        'YTickLabel',{'1','2','3','4','5', '6','7','8','9','10','11','12','13'},...
        'TickLength',[0 0]);

% colormap('jet');
% imagesc(D)
% colorbar;

%hmo  = HeatMap(D)

%surf(D)

%Xlabel = [1,2,3,4,5,6,7,8,9,10,11,12,13];

%addXLabel(hmo, 'Actual Drug Label');
%addYLabel(hmo, 'Predictions');


